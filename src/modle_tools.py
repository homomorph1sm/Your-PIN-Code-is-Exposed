import os
import matplotlib
import paddlex as pdx
from paddlex.det import transforms
import cv2
import math
import json


def train_model():
    train_transforms = transforms.Compose([
        transforms.MixupImage(mixup_epoch=250),
        transforms.RandomDistort(),
        transforms.RandomExpand(),
        transforms.RandomCrop(),
        transforms.Resize(target_size=608, interp='RANDOM'),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(target_size=608, interp='CUBIC'),
        transforms.Normalize(),
    ])

    dataset_path = 'fingertip_v1'

    train_dataset = pdx.datasets.VOCDetection(
        data_dir=dataset_path,
        file_list=dataset_path + '/train_list.txt',
        label_list=dataset_path + '/labels.txt',
        transforms=train_transforms,
        shuffle=True)

    eval_dataset = pdx.datasets.VOCDetection(
        data_dir=dataset_path,
        file_list=dataset_path + '/train_list.txt',
        label_list=dataset_path + '/labels.txt',
        transforms=eval_transforms)

    num_classes = len(train_dataset.labels)
    model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')
    model.train(
        num_epochs=140,
        train_dataset=train_dataset,
        train_batch_size=8,
        eval_dataset=eval_dataset,
        learning_rate=0.000125,
        lr_decay_epochs=[210, 240],
        save_interval_epochs=20,
        save_dir='output/yolov3_darknet53',
        use_vdl=True)
    return model


def load_model():
    model = pdx.load_model('./model_v1')
    return model


def use_model(model, pic):
    """
    doc: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#predict
    :param model:
    :param pic:
    :return:
    """
    image_name = 'data/1.jpg'
    result = model.predict(image_name)
    print("result:", result)
    for i in result:
        print(i['category'])
        print(i['bbox'])
        print(i['score'])

    pdx.det.visualize(image_name, result, threshold=0.5, save_dir='./data/')


def use_model_for_video(model_path, video_path):
    print("--------------------START--------------------")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
    os.environ['CPU_NUM'] = '4' 
    model = pdx.load_model(model_path)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(''.join(video_path.split('.')[:-1]) + '_out.mp4', fourcc, fps, size)
    last_pos = [0, 0, 0, 0]
    i = 0
    v_list = [0]
    result_file = open(''.join(video_path.split('.')[:-1]) + '_out.json', 'w')
    while True:
        ret, frame = cap.read()
        if ret:
            result = model.predict(frame)
            j = json.dumps(result)
            result_file.write(j)
            result_file.write(os.linesep)

            bbox = get_bbox(result, 'tag')
            if bbox:
                crop_img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                cv2.imwrite("cut/cut%d.jpg" % i, crop_img)

                v = int(compute_instantaneous_velocity(last_pos, bbox))
                v_list.append(v)
                last_pos = bbox
            else:
                v_list.append(v_list[-1])
            new_frame = pdx.det.visualize(frame, result, threshold=0.1, save_dir=None)
            new_frame = cv2.putText(new_frame, str(v_list[-1]), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(new_frame)
            # cv2.imshow("video", new_frame)
            print(i)
            i += 1
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
        else:
            break
    result_file.close()
    # cap.release()
    # cv2.destroyAllWindows()

    # model = my_yolo3.load_model()
    # result = model.predict(image_name)
    # pdx.det.visualize(image_name, result, threshold=0.5, save_dir='./data/')
    print("-------------------- END --------------------")
    return v_list


def get_bbox(result, tag):
    for r in result:
        if r['category'] == tag:
            bbox = [int(x) for x in r['bbox']]
            break
    else:
        bbox = []
    return bbox


def compute_instantaneous_velocity(last_pos, now_pos):
    last_point = [(last_pos[0] + last_pos[2])/2, (last_pos[1] + last_pos[3])/2]
    now_point = [(now_pos[0] + now_pos[2])/2, (now_pos[1] + now_pos[3])/2]
    return math.sqrt((now_point[0] - last_point[0])**2 + (now_point[1] - last_point[1])**2)*30
