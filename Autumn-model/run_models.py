import argparse
import autumn
import cv2
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Runner for chauffeur')
    parser.add_argument('--target_folder', help='Root folder of image files')
    parser.add_argument('--target_frame', help='{private, public, all}')
    parser.add_argument('--target_model', help='{autumn}', default='autumn')
    args = parser.parse_args()

    # load CH2_final_evaluation.csv../udacity_data/
    with open('./CH2_final_evaluation.csv', 'r') as f:
        final_eval = f.readlines()
    public_frames = list()
    private_frames = list()
    for line in final_eval:
        if 'frame_id' in line:
            continue
        frame_id, steering_angle, is_public = line.split(',')
        if is_public.strip() == '1':
            public_frames.append(frame_id)
        else:
            private_frames.append(frame_id)

    # load the DNN model
    if args.target_model == 'autumn':
        dnn = autumn.get_predictor()
    # load images
    target_folder = args.target_folder
    images = list()
    for root, sub_dirs, files in os.walk(target_folder):
        for f in files:
            if '.jpg' in f or '.png' in f:
                images.append((root, f))
    images.sort(key=lambda x: x[1])

    # start prediction
    result_file = 'prediction_{}_{}.txt'.format(args.target_frame, args.target_model)
    with open(result_file, 'w') as f:
        count = 0
        for image in images:
            img_path, img_file = image
            frame_id = os.path.splitext(img_file)[0]

            # Skip if the target_frame is not private (or public)
            if (args.target_frame == 'private' and frame_id in public_frames) or \
                    (args.target_frame == 'public' and frame_id in private_frames):
                continue

            if args.target_model != 'rambo':
                image_pred = cv2.imread(os.path.join(img_path, img_file))
            else:
                image_pred = os.path.join(img_path, img_file)
            f.write('{},{}\n'.format(frame_id, dnn(image_pred)))
            count += 1
            if count % 1000 == 0:
                print('{} images processed ...'.format(count))
        print 'Total processed images:', count
    print result_file, 'has been updated. Done.'