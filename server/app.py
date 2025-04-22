from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import mediapipe as mp # mp đã được import
import cv2
from PIL import Image
import math
import base64
import io # Để xử lý dữ liệu ảnh dạng bytes
import os # Để kiểm tra sự tồn tại của file

# --- Cấu hình ---
# Đảm bảo đường dẫn đến file mô hình và label encoder là chính xác
# so với vị trí bạn chạy file app.py.
# Ví dụ: Nếu bạn đặt file app.py cùng cấp với thư mục 'models', thì giữ nguyên.
# Nếu không, hãy điều chỉnh đường dẫn (ví dụ: '../models/...') hoặc dùng đường dẫn tuyệt đối.
model_path = 'model/gesture_model_json.h5'
label_encoder_path = 'model/label_encoder.pkl'
imageSize = 200 # Không trực tiếp sử dụng trong API xử lý, giữ lại để tham khảo nếu cần

# --- Khởi tạo các đối tượng toàn cục ---
# Các đối tượng này sẽ được load/khởi tạo một lần khi server bắt đầu
loaded_model = None
loaded_label_encoder = None
hands_detector = None # Đối tượng MediaPipe hands

# Định nghĩa mp_hands từ import mediapipe
# Đây là biến toàn cục cần được truy cập trong load_resources
mp_hands = mp.solutions.hands

# --- Khởi tạo ứng dụng Flask ---
app = Flask(__name__)
# Cho phép Cross-Origin Resource Sharing (CORS) để ứng dụng di động có thể kết nối
# Trong môi trường production, bạn nên cấu hình CORS chặt chẽ hơn để chỉ cho phép
# các nguồn đáng tin cậy kết nối.
CORS(app)

# --- Hàm tải tài nguyên (mô hình, label encoder, MediaPipe) ---
# Hàm này sẽ chạy một lần trước yêu cầu đầu tiên đến server
@app.before_request
def load_resources():
    """Tải mô hình, label encoder và khởi tạo MediaPipe Hand."""
    # Thêm mp_hands vào dòng khai báo global để khắc phục NameError
    global loaded_model, loaded_label_encoder, hands_detector, mp_hands

    # Chỉ tải nếu chưa được tải
    if loaded_model is None or loaded_label_encoder is None or hands_detector is None:
        print(">>> Debug: Đang tải mô hình và các tài nguyên...") # Thêm debug log
        try:
            # Kiểm tra sự tồn tại của file trước khi tải
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {model_path}")
            if not os.path.exists(label_encoder_path):
                raise FileNotFoundError(f"Không tìm thấy file label encoder tại: {label_encoder_path}")

            # Tải mô hình TensorFlow
            print(f">>> Debug: Đang tải mô hình từ {model_path}") # Thêm debug log
            # compile=False được dùng khi bạn chỉ cần load để inference
            loaded_model = tf.keras.models.load_model(model_path, compile=False)
            print(">>> Debug: Mô hình TensorFlow đã tải.") # Thêm debug log


            # Tải label encoder
            print(f">>> Debug: Đang tải label encoder từ {label_encoder_path}") # Thêm debug log
            with open(label_encoder_path, 'rb') as le_file:
                # Đảm bảo scikit-learn đã được cài đặt để pickle.load hoạt động
                loaded_label_encoder = pickle.load(le_file)
            print(">>> Debug: Label encoder đã tải.") # Thêm debug log


            # Khởi tạo đối tượng MediaPipe Hands
            print(">>> Debug: Đang khởi tạo MediaPipe Hands") # Thêm debug log
            # Lỗi NameError xảy ra ở dòng dưới trước đó
            hands_detector = mp_hands.Hands(
                static_image_mode=True, # Chế độ xử lý ảnh tĩnh
                max_num_hands=1,        # Chỉ phát hiện 1 tay
                min_detection_confidence=0.5 # Ngưỡng tin cậy để phát hiện tay
            )
            print(">>> Debug: MediaPipe Hands đã khởi tạo.") # Thêm debug log


            print(">>> Debug: Tất cả tài nguyên đã tải thành công. Kết thúc load_resources()") # Thêm debug log


        except FileNotFoundError as fnf_error:
            print(f">>> Lỗi File Not Found: {fnf_error}") # Thêm debug log
            print("Vui lòng kiểm tra lại đường dẫn tới mô hình và label encoder.")
            # Đặt lại các biến thành None để các request sau biết là tài nguyên chưa sẵn sàng
            loaded_model = loaded_label_encoder = hands_detector = None
        except Exception as e:
            print(f">>> Lỗi khi tải tài nguyên: {e}") # Thêm debug log
            import traceback
            traceback.print_exc()
             # Đặt lại các biến thành None nếu có lỗi tải
            loaded_model = loaded_label_encoder = hands_detector = None


# --- Các hàm xử lý logic (được điều chỉnh từ script gốc) ---

# Hàm calculate_physical_size được giữ lại nhưng không trực tiếp ảnh hưởng đến đầu vào mô hình
# nếu mô hình chỉ dùng landmark. Tuy nhiên, nó vẫn hữu ích cho việc tính crop_box.
def calculate_physical_size(image, hand_landmarks):
    """
    Tính toán kích thước vật lý của bàn tay trong ảnh và bounding box.
    Được điều chỉnh để làm việc với ảnh dạng NumPy array.
    """
    # Image đầu vào là NumPy array (ảnh đã đọc bằng cv2)
    img_height, img_width = image.shape[:2]

    # Lấy bounding box của bàn tay
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')

    for landmark in hand_landmarks.landmark:
        # landmark.x, landmark.y, landmark.z là tọa độ đã chuẩn hóa [0, 1]
        # Chuyển về tọa độ pixel
        x, y = int(landmark.x * img_width), int(landmark.y * img_height)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    # Tính khoảng cách pixel giữa các điểm JOINTS cụ thể để ước lượng kích thước
    # Ở đây dùng khoảng cách giữa INDEX_FINGER_MCP và PINKY_MCP
    try:
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

        index_x, index_y = int(index_mcp.x * img_width), int(index_mcp.y * img_height)
        pinky_x, pinky_y = int(pinky_mcp.x * img_width), int(pinky_mcp.y * img_height)

        # Tính khoảng cách Euclid trong pixel
        pixel_distance = math.sqrt((index_x - pinky_x)**2 + (index_y - pinky_y)**2)

        # Tránh chia cho 0 hoặc số rất nhỏ
        if pixel_distance < 1.0:
            # Nếu khoảng cách quá nhỏ, có thể do điểm landmark không chính xác.
            # Fallback: dùng chiều rộng của bounding box để ước lượng cm/pixel.
            hand_width_pixels = x_max - x_min
            cm_per_pixel = 8.4 / hand_width_pixels if hand_width_pixels > 0 else 0.0
        else:
             # Ước tính tỉ lệ cm trên pixel dựa trên giả định chiều rộng bàn tay người lớn
            cm_per_pixel = 8.4 / pixel_distance

    except Exception as e:
        # Xử lý nếu có lỗi khi truy cập landmark hoặc tính toán
        print(f"Lỗi khi tính toán kích thước vật lý: {e}")
        cm_per_pixel = 0.0 # Đặt về 0 để tránh lỗi

     # Đảm bảo bounding box hợp lệ trước khi trả về
    if x_min > x_max or y_min > y_max:
         print("Warning: Bounding box không hợp lệ.")
         # Trả về giá trị mặc định hoặc báo lỗi tùy vào yêu cầu
         return cm_per_pixel, (0, 0, image.shape[1], image.shape[0]) # Trả về toàn ảnh

    return cm_per_pixel, (x_min, y_min, x_max, y_max)


# Hàm crop_hand_with_padding và visualize_detection từ script gốc không được sử dụng
# trong luồng xử lý chính của API /predict vì API chỉ cần trả về kết quả dự đoán.
# Bạn có thể giữ lại chúng ở đây nếu muốn để dùng cho mục đích debug hoặc mở rộng sau này.
# def crop_hand_with_padding(...): ...
# def visualize_detection(...): ...


def extract_landmarks(hand_landmarks, image_width, image_height):
    """
    Trích xuất tọa độ landmark đã chuẩn hóa từ phát hiện tay.
    Trả về danh sách phẳng (flattened list) các tọa độ (x, y, z) cho mỗi landmark.
    """
    landmark_coords = []

    # landmarks trả về bởi MediaPipe đã được chuẩn hóa theo kích thước ảnh gốc [0, 1]
    for landmark in hand_landmarks.landmark:
        # Chỉ cần thêm tọa độ x, y, z đã chuẩn hóa
        # Đảm bảo chuyển đổi sang kiểu dữ liệu phù hợp nếu cần, mặc định là float
        landmark_coords.extend([landmark.x, landmark.y, landmark.z])

    return landmark_coords


# Đây là hàm chính thực hiện logic xử lý ảnh và dự đoán,
# tương tự như phần cốt lõi của script gốc nhưng nhận bytes ảnh làm đầu vào.
def predict_gesture_from_image_data(image_data_bytes):
    """
    Xử lý dữ liệu ảnh dạng bytes, phát hiện tay, trích xuất landmark và dự đoán thủ ngữ.

    Args:
        image_data_bytes: Dữ liệu ảnh dưới dạng bytes.

    Returns:
        Một dictionary chứa kết quả dự đoán (label, confidence)
        hoặc thông báo lỗi/không phát hiện tay.
    """
    global loaded_model, loaded_label_encoder, hands_detector

    # Kiểm tra xem tài nguyên đã được tải chưa
    if loaded_model is None or loaded_label_encoder is None or hands_detector is None:
         print("Lỗi: Mô hình hoặc tài nguyên khác chưa được tải.")
         return {'error': 'Server chưa sẵn sàng. Vui lòng thử lại sau.'}

    try:
        # --- BƯỚC 1: Đọc và tiền xử lý ảnh ---
        # Tương tự hàm preprocess_image trong script gốc, nhưng từ bytes
        np_arr = np.frombuffer(image_data_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Đọc ảnh

        if img is None:
             print("Lỗi: Không thể giải mã dữ liệu ảnh.")
             return {'error': 'Không thể giải mã dữ liệu ảnh được gửi.'}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Chuyển BGR sang RGB
        img_height, img_width = img_rgb.shape[:2]

        # --- BƯỚC 2: Phát hiện tay và trích xuất Landmark bằng MediaPipe ---
        # Tương tự phần dùng mp_hands trong script gốc
        # Sử dụng đối tượng hands_detector đã được khởi tạo trong load_resources
        results = hands_detector.process(img_rgb)

        # --- BƯỚC 3: Xử lý kết quả từ MediaPipe và chuẩn bị dữ liệu cho mô hình ---
        # Tương tự phần kiểm tra results.multi_hand_landmarks trong script gốc
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0] # Lấy tay đầu tiên

            # Tính toán crop_box (tùy chọn, không dùng làm input model)
            # Cần truy cập mp_hands ở đây, nó được định nghĩa toàn cục
            cm_per_pixel, crop_box = calculate_physical_size(img_rgb, hand_landmarks)

            # Trích xuất tọa độ landmark đã chuẩn hóa (INPUT cho mô hình)
            # Tương tự hàm extract_landmarks trong script gốc
            landmark_coords = extract_landmarks(hand_landmarks, img_width, img_height)

            # Chuẩn bị dữ liệu landmark cho mô hình (dạng NumPy array)
            processed_landmarks = np.array([landmark_coords], dtype=np.float32) # Shape (1, 63)

            # Kiểm tra kích thước input
            if processed_landmarks.shape == (1, 63):
                # --- BƯỚC 4: Dự đoán bằng mô hình TensorFlow ---
                # Tương tự phần model.predict trong script gốc
                prediction = loaded_model.predict(processed_landmarks)

                # --- BƯỚC 5: Hậu xử lý kết quả và trả về ---
                # Tương tự phần lấy argmax và inverse_transform trong script gốc
                predicted_class_index = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class_index]) # Chuyển sang float

                # Kiểm tra xem label_encoder đã được tải chưa trước khi sử dụng
                if loaded_label_encoder is None:
                     print("Lỗi: Label encoder chưa được tải.")
                     return {'error': 'Label encoder chưa sẵn sàng.'}

                predicted_label = loaded_label_encoder.inverse_transform([predicted_class_index])[0]

                # Trả về kết quả
                return {
                    'predicted_label': str(predicted_label), # Chuyển sang string
                    'confidence': confidence,
                    # Có thể trả về landmark_coords hoặc crop_box nếu cần visualization trên app
                    # 'landmarks': landmark_coords,
                    # 'crop_box': crop_box
                }
            else:
                 print(f"Lỗi xử lý: Số lượng landmark không đúng. Nhận shape {processed_landmarks.shape}, mong đợi (1, 63).")
                 return {'error': f'Lỗi xử lý dữ liệu landmark.'}
        else:
            # Không phát hiện tay
            print("Không phát hiện tay trong ảnh.")
            return {'message': 'Không phát hiện tay trong ảnh. Vui lòng thử lại.'}

    except Exception as e:
        # Bắt các lỗi khác
        print(f"Lỗi chung khi xử lý ảnh và dự đoán: {e}")
        import traceback
        traceback.print_exc()
        return {'error': f'Đã xảy ra lỗi trong quá trình xử lý: {e}'}


# --- Định nghĩa Endpoint API ---

@app.route('/predict', methods=['POST'])
def predict_gesture_api():
    """
    Endpoint API để nhận ảnh (Base64 encoded trong JSON) và trả về dự đoán thủ ngữ.
    """
    # Kiểm tra request
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'Body request không hợp lệ. Mong đợi JSON với key "image".'}), 400

    try:
        # Lấy và giải mã ảnh Base64
        base64_image_string = request.json['image']
        # Loại bỏ tiền tố "data:image/png;base64," nếu có
        if "," in base64_image_string:
             base64_image_string = base64_image_string.split(",")[1]

        image_data_bytes = base64.b64decode(base64_image_string)

        # Gọi hàm xử lý chính
        results = predict_gesture_from_image_data(image_data_bytes)

        # Trả về kết quả JSON
        if results and 'error' not in results:
             # Thành công hoặc có thông báo (ví dụ: không phát hiện tay)
             return jsonify(results), 200
        elif results and 'message' in results:
             # Trường hợp không phát hiện tay
             return jsonify(results), 200
        else:
             # Lỗi xảy ra trong predict_gesture_from_image_data
             return jsonify(results), 500

    except Exception as e:
        # Bắt các lỗi trước khi gọi hàm xử lý
        print(f"Lỗi API chung: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Đã xảy ra lỗi không mong muốn tại API: {e}'}), 500

# Endpoint gốc
@app.route('/')
def index():
    """Endpoint đơn giản để kiểm tra server có phản hồi không."""
    return "API Nhận Diện Thủ Ngữ đang chạy!"

# --- Khởi chạy Server Flask ---
if __name__ == '__main__':
    # Khi chạy file này trực tiếp
    print("Bắt đầu khởi chạy server Flask...")
    # load_resources() # Có thể gọi ở đây hoặc dựa vào @app.before_request
    app.run(host='0.0.0.0', port=5000, debug=True) # Chạy server development