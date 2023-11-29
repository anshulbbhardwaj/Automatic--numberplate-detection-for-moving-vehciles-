import ast
import string
import cv2 as cv
import easyocr
from ultralytics import YOLO

video = cv.VideoCapture(r"C:\Users\piyus\Downloads\170609_A_Delhi_026.mp4")
model = YOLO(r"C:\Users\piyus\PycharmProjects\pythonProject7\runs\detect\train29\weights\best.pt")
ret = True
frame_number = -1
results = {}
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
# characters that can easily be confused can be
# verified by their location - an `O` in a place
# where a number is expected is probably a `0`
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def license_complies_format(text):
    # True if the license plate complies with the format, False otherwise.
    if len(text) <= 8:
        return False
    if len(text) == 9:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
                (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
                (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    2] in dict_char_to_int.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    3] in dict_char_to_int.keys()) and \
                (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
                (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    5] in dict_char_to_int.keys()) and \
                (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    6] in dict_char_to_int.keys()) and \
                (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    7] in dict_char_to_int.keys()) and \
                (text[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[8] in dict_char_to_int.keys()):
            return True

    elif len(text) == 10:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
                (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
                (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    2] in dict_char_to_int.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    3] in dict_char_to_int.keys()) and \
                (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
                (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
                (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    6] in dict_char_to_int.keys()) and \
                (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    7] in dict_char_to_int.keys()) and \
                (text[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    8] in dict_char_to_int.keys()) and \
                (text[9] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[9] in dict_char_to_int.keys()):
            return True
    else:
        return False



def format_license(text):
    license_plate_ = ''
    if len(text) == 10:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 9: dict_char_to_int, 4: dict_int_to_char,
                   5: dict_int_to_char, 6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int,
                   2: dict_char_to_int, 3: dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]


    else:
        mapping1 = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char,
                    6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int,
                    2: dict_char_to_int, 3: dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            if text[j] in mapping1[j].keys():
                license_plate_ += mapping1[j][text[j]]
            else:
                license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        # verify that text is conform to a standard license plate
        if license_complies_format(text):
            # bring text into the default license plate format
            return format_license(text), score

    return None, None


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(
            'frame_number', 'track_id', 'car_bbox', 'car_bbox_score',
            'license_plate_bbox', 'license_plate_bbox_score', 'license_plate_number',
            'license_text_score'))

        for frame_number in results.keys():
            for track_id in results[frame_number].keys():
                print(results[frame_number][track_id])
                if 'car' in results[frame_number][track_id].keys() and \
                        'license_plate' in results[frame_number][track_id].keys() and \
                        'number' in results[frame_number][track_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                        frame_number,
                        track_id,
                        '[{} {} {} {}]'.format(
                            results[frame_number][track_id]['car']['bbox'][0],
                            results[frame_number][track_id]['car']['bbox'][1],
                            results[frame_number][track_id]['car']['bbox'][2],
                            results[frame_number][track_id]['car']['bbox'][3]
                        ),
                        results[frame_number][track_id]['car']['bbox_score'],
                        '[{} {} {} {}]'.format(
                            results[frame_number][track_id]['license_plate']['bbox'][0],
                            results[frame_number][track_id]['license_plate']['bbox'][1],
                            results[frame_number][track_id]['license_plate']['bbox'][2],
                            results[frame_number][track_id]['license_plate']['bbox'][3]
                        ),
                        results[frame_number][track_id]['license_plate']['bbox_score'],
                        results[frame_number][track_id]['license_plate']['number'],
                        results[frame_number][track_id]['license_plate']['text_score'])
                    )
        f.close()


# read the 10 first frames
while ret:
    frame_number += 1
    ret, frame = video.read()

    if ret and frame_number < 500:
        results[frame_number] = {}
        # vehicle detector
        detections = model.predict(frame)[0]

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score = detection
            plates = [[x1, y1, x2, y2, track_id, score]]
            for bbox in plates:
                print(bbox)
                roi = frame[int(y1):int(y2), int(x1):int(x2)]

                plate_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)


                # posterize
                _, plate_treshold = cv.threshold(plate_gray, 64, 255, cv.THRESH_BINARY_INV)

                np_text, np_score = read_license_plate(plate_gray)

                if np_text is not None:
                    results[frame_number][track_id] = {
                        'car': {
                            'bbox': [x1, y1, x2, y2],
                            'bbox_score': score
                        },
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'bbox_score': score,
                            'number': np_text,
                            'text_score': np_score
                        }
                    }

write_csv(results, './results.csv')

video.release()

print("-------------------------------------")
print(results)

print(results[130])


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=6, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
    cv.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
    cv.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
    cv.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
    cv.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


video = cv.VideoCapture(r"C:\Users\piyus\Downloads\170609_A_Delhi_026.mp4")

# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, size)

# reset video before you re-run cell below
frame_number = -1
video.set(cv.CAP_PROP_POS_FRAMES, 0)

ret = True

while ret:
    ret, frame = video.read()
    frame_number += 1
    if ret:
        df_ = results[results[frame_number] == frame_number]
        for index in range(len(df_)):
            # draw car
            x1, y1, x2, y2 = ast.literal_eval(
                df_.iloc[index]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

            draw_border(
                frame, (int(x1), int(y1)),
                (int(x2), int(y2)), (0, 255, 0),
                12, line_length_x=200, line_length_y=200)

            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(
                df_.iloc[index]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(
                    ' ', ','))

            # region of interest
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            cv.rectangle(roi, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 6)

            # write detected number
            (text_width, text_height), _ = cv.getTextSize(
                df_.iloc[index]['license_plate_number'],
                cv.FONT_HERSHEY_SIMPLEX,
                2,
                6)

            cv.putText(
                frame,
                df_.iloc[index]['license_plate_number'],
                (int((x2 + x1 - text_width) / 2), int(y1 - text_height)),
                cv.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                6
            )

        out.write(frame)
        frame = cv.resize(frame, (1280, 720))

out.release()
video.release()
print('hi')
