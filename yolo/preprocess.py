import cv2
import numpy as np
import os
import glob

def process_image(image_path, output_path=None, plot=False):
    """
    將圖片進行二值化處理、消除胡椒雜訊並執行透視變換。

    :param image_path: 輸入圖片路徑
    :param output_path: 輸出圖片路徑（可選）
    :param plot: 是否顯示處理過程
    :return: 處理後的影像
    """
    # 讀取影像
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"無法讀取影像: {image_path}")

    # 獲取影像尺寸
    height, width = frame.shape[:2]

    # 轉換為灰階影像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 二值化處理
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # 消除胡椒雜訊（形態學開運算）
    kernel = np.ones((5, 5), np.uint8)  # 定義 3x3 的核
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if plot:
        # 顯示原始影像、二值化影像和去雜訊後影像
        cv2.imshow("Original Image", frame)
        cv2.imshow("Binary Image", binary)
        cv2.imshow("Cleaned Binary Image", binary_cleaned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 儲存結果影像
    if output_path:
        cv2.imwrite(output_path, binary_cleaned)

    return binary_cleaned

def process_folder(input_folder, output_folder, plot=False):
    """
    處理整個資料夾中的所有圖片

    :param input_folder: 輸入資料夾路徑
    :param output_folder: 輸出資料夾路徑
    :param plot: 是否顯示處理過程
    """
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 支持的圖片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

    processed_count = 0

    # 處理所有圖片
    for extension in image_extensions:
        pattern = os.path.join(input_folder, extension)
        image_files = glob.glob(pattern)

        for image_path in image_files:
            try:
                # 生成輸出檔案名
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_processed{ext}"
                output_path = os.path.join(output_folder, output_filename)
                print(f"處理中: {filename}")
                # 處理圖片
                process_image(image_path, output_path, plot=plot)
                processed_count += 1
                print(f"已處理: {filename} -> {output_filename}")
            except Exception as e:
                print(f"處理 {filename} 時出錯: {str(e)}")
    print(f"\n處理完成！共處理 {processed_count} 張圖片")
    print(f"輸出位置: {output_folder}")

def main():
    # 設定輸入和輸出資料夾路徑
    input_folder = "./images_road_signs"    # 替換為您的輸入資料夾
    output_folder = "./images_road_signs_after"  # 替換為您的輸出資料夾
    # 處理整個資料夾
    process_folder(input_folder, output_folder, plot=False)

if __name__ == "__main__":
    main()