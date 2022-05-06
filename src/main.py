import video_tools
import dataset_tools
import modle_tools
import analysis_tools


def video_preprocessing():
    video_folder = '/home/zono/Documents/paper_code/data/raw_video'
    picture_folder = '/home/zono/Documents/paper_code/data/raw_picture'
    dataset_tools.mkdir(picture_folder)
    video_list1 = ['1_1.mp4', '1_2.mp4', '1_3.mp4', '1_4.mp4', '1_5.mp4']
    video_list2 = ['2_1.mp4', '2_2.mp4']

    for n1 in video_list1:
        video_tools.extract_frames(video_folder + '/' + n1, picture_folder, 5)
    for n2 in video_list2:
        video_tools.extract_frames(video_folder + '/' + n2, picture_folder, 5)


def dataset_preprocessing():
    dataset_path_list = [
        '/home/zono/Documents/paper_code/data/raw_dataset/Dataset_p1',
        '/home/zono/Documents/paper_code/data/raw_dataset/Dataset_p2'
    ]
    label_list = ['tag', 'n1', 'n3', 'n8']
    train_test_rate = 0.8

    dataset_tools.add_txt(dataset_path_list, label_list, train_test_rate)


def video_processing():
    model_path = '/home/zono/Documents/paper_code/yolo_model/p1_model'
    video_path = '/home/zono/Documents/paper_code/data/raw_video/3_1.mp4'
    v_list = modle_tools.use_model_for_video(model_path, video_path)


def compute_processing():
    video_path = '/home/zono/Documents/paper_code/data/raw_video/1_4.mp4'
    yolo_result_path = '/home/zono/Documents/paper_code/data/raw_video/1_4_out.json'
    out_path = '/home/zono/Documents/paper_code/data/raw_video/1_4_show_with_flow.mp4'
    system = analysis_tools.MatchingSystem(video_path, yolo_result_path, '')

if __name__ == '__main__':
    video_processing()