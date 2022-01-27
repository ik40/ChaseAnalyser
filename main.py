from pytube import YouTube
import sys
import cv2
import numpy as np
from itertools import islice
from csv import reader
# import pytesseract as pt


class ChaseAnalyser():
    GREEN = 'green'
    LBLUE = 'lightblue'
    DBLUE = 'darkblue'
    RED = 'red'
    BLACK = 'black'
    QBLUE = 'question blue'
    ChaserRight = 'Chaser gets answer right'
    ChaserWrong = 'Chaser gets answer wrong'
    ContestantRight = 'Contestant gets answer right'
    ContestantWrong = 'Contestant gets answer wrong'
    Neither = 'Neither get answer right'
    Both = 'Both get answer right'
    ARight = 'A correct'
    BRight = 'B correct'
    CRight = 'C correct'
    Placeholder = 'Placeholder'
    TopChoice = 'Top choice'
    MidChoice = 'Middle choice'
    LowChoice = 'Low choice'

    # download videos (high res)
    def download_videos(self):
        with open(sys.argv[1]) as f:
            lines = f.readlines()
        for url in lines:
            yt = YouTube(url)
            streams = yt.streams.filter(progressive=True)
            itag = streams[-1].itag
            stream = yt.streams.get_by_itag(itag)
            stream.download()

    # get frames from the videos
    def get_frames(self, stream):
        video = cv2.VideoCapture(stream)
        i = 0
        repeater = 0
        last_frame = False
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            else:
                frames = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if i % 30 == 0 and i >= 70000:
                    # get question boxes (with no options) to use as signal to get previous question's last frame
                    if x.strip_blue(590, 230, 1050, frames):
                        if last_frame is not None:
                            cv2.imwrite(F"{i}.png", last_frame)
                            continue
                    elif x.area(640, 230, 1050, 25, frames, ChaseAnalyser.GREEN):
                        last_frame = frame
                i+=1

                    # get question boxes (with options)
                    # if x.strip(534, 230, 1050, frames):
                    #     if i - repeater < 50:
                    #         continue
                    #     else:
                    # # get question boxes with a GREEN option
                    #         if x.area(640, 230, 1050, 25, frames, ChaseAnalyser.GREEN):
                    #             # print('green')
                    #             cv2.imwrite(F"questions/green113{i}.png", frame)
                    #             repeater = i
                i += 1
        video.release()
        cv2.destroyAllWindows()

    def strip_blue(self, pixely, pixelx1, pixelx2, frame):
        count_blue = 1
        count_other = 1
        for x in range(pixelx1, pixelx2):
            col = self.colour_checker(frame[pixely][x])
            if col == ChaseAnalyser.QBLUE:
                count_blue += 1
            else:
                count_other += 1
        if (count_blue / count_other) > 0.3:
            return True

    def strip_green(self, pixely, pixelx1, pixelx2, frame):
        # frame = cv2.imread(frame)
        count_green = 1
        for x in range(pixelx1, pixelx2):
            col = self.colour_checker(frame[pixely][x])
            # print(frame[pixely][x])
            if col == ChaseAnalyser.GREEN:
                count_green += 1
        # print(count_green)
        if count_green >= 50:
            return True

    def area(self, pixely1, pixelx1, pixelx2, offset, frame, colour):
        for y in range(pixely1, (pixely1+offset), 2):
            if colour == ChaseAnalyser.GREEN:
                strip1 = self.strip_green(y, pixelx1, pixelx2, frame)
            else:
                strip1 = self.strip_blue(y, pixelx1, pixelx2, frame)
            if strip1:
                return True

    def masking(self, img_path):
        # read in image
        path = 'tests/sc7.png'
        nm = 'sc7'
        img = cv2.imread(path)

        # convert to rgb as im.. functions use bgr
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # set range for blue detection
        light_blue = (90, 70, 50)
        dark_blue = (128, 255, 255)
        # create mask for blue detection using bitwise AND
        mask_blue = cv2.inRange(rgb_img, light_blue, dark_blue)
        result_blue = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_blue)

        # make green mask and apply using bitwise AND
        green = (40, 40, 40)
        dark_green = (70, 255, 255)
        mask_green = cv2.inRange(rgb_img, green, dark_green)
        result_green = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_green)

        # make red mask and apply using bitwise AND
        lower_red = np.array([139, 0, 0])
        upper_red = np.array([255, 204, 203])
        mask_red = cv2.inRange(rgb_img, lower_red, upper_red)
        result_red = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_red)

        # make white mask and get everything outside of it
        lower_white = np.array([0, 0, 168])
        upper_white = np.array([172, 111, 255])
        mask_white = cv2.inRange(rgb_img, lower_white, upper_white)
        result_white = cv2.bitwise_not(rgb_img, rgb_img, mask=mask_white)

        # read masks
        mask1 = result_blue
        mask2 = result_white
        mask3 = result_red
        mask4 = result_green

        # add masks
        result = (mask1 + mask3 + mask4)
        result = result.clip(0, 255).astype("uint8")

        # save results
        cv2.imwrite(F"mask{nm}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


    def colour_checker(self, pixel):
        if 0 <= pixel[0] <= 20 and 156 <= pixel[1] <= 233 and 0 <= pixel[2] <= 20:
            return ChaseAnalyser.GREEN
        # elif 139 <= pixel[0] <= 149 and 50 <= pixel[1] <= 80 and 50 <= pixel[2] <= 83:
        #     return ChaseAnalyser.RED
        elif 145 <= pixel[0] <= 256 and 0 <= pixel[1] <= 80 and 0 <= pixel[2] <= 83:
            return ChaseAnalyser.RED
        # elif 42 <= pixel[0] <= 70 and 75 <= pixel[1] <= 112 and 105 <= pixel[2] <= 132:
        #     return ChaseAnalyser.DBLUE
        elif 0 <= pixel[0] <= 10 and 0 <= pixel[1] <= 1 and 190 <= pixel[2] <= 256:
            return ChaseAnalyser.DBLUE
        # elif 128 <= pixel[0] <= 132 and 155 <= pixel[1] <= 208 and 200 <= pixel[2] <= 239:
        #     return ChaseAnalyser.LBLUE
        elif 4 <= pixel[0] <= 28 and 160 <= pixel[1] <= 211 and 218 <= pixel[2] <= 250:
            return ChaseAnalyser.LBLUE
        elif 119 <= pixel[0] <= 170 and 120 <= pixel[1] <= 180 and 180 <= pixel[2] <= 245:
            return ChaseAnalyser.QBLUE
        else:
            return ChaseAnalyser.BLACK

    def logic(self, img_path):
        analysis = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        check = analysis[1370]
        seq = []
        current = []
        for index, item in islice(enumerate(check), 900, 2163):
            # skips black pixels
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                continue
            else:
                temp = self.colour_checker(item)
                if temp == current or temp == ChaseAnalyser.BLACK:
                    continue
                else:
                    current = temp
                    seq.append(current)
        #     print(index)
        print(seq)
        return seq

    def inference(self, seq):
        temp = []
        final = []
        right_answer = ChaseAnalyser.Placeholder

        for i in seq:
            if i == ChaseAnalyser.RED:
                continue
            else:
                temp.append(i)
        # print(temp)

        if len(temp) == 2:
            contestant_right = ChaseAnalyser.ContestantRight
            if temp[0] == ChaseAnalyser.GREEN:
                right_answer = ChaseAnalyser.ARight
                # print('A')
                final.append(contestant_right)
            else:
                right_answer = ChaseAnalyser.CRight
                # print('B')
                final.append(contestant_right)

        elif len(temp) == 3:
            contestant_right = ChaseAnalyser.ContestantWrong
            if temp[0] == ChaseAnalyser.DBLUE and temp[1] == ChaseAnalyser.DBLUE or temp[1] == ChaseAnalyser.DBLUE and \
                    temp[2] == ChaseAnalyser.DBLUE:
                contestant_right = ChaseAnalyser.ContestantRight
                final.append(contestant_right)
            else:
                if temp[0] == ChaseAnalyser.GREEN:
                    right_answer = ChaseAnalyser.ARight
                    print('C')
                elif temp[1] == ChaseAnalyser.GREEN:
                    print('D')
                    right_answer = ChaseAnalyser.BRight
                    if temp[0] == ChaseAnalyser.DBLUE and temp[2] == ChaseAnalyser.DBLUE:
                        contestant_right = ChaseAnalyser.ContestantRight
                else:
                    right_answer = ChaseAnalyser.CRight
                    contestant_right = ChaseAnalyser.ContestantWrong
                    print('E')
                final.append(contestant_right)
        else:
            print('Something is terribly wrong..')

        reds = 0
        for i in seq:
            if i == ChaseAnalyser.RED:
                reds += 1
        if reds == 2 and ChaseAnalyser.BRight:
            chaser_right = ChaseAnalyser.ChaserRight
            final.append(chaser_right)
        elif reds == 1:
            if right_answer == ChaseAnalyser.ARight and seq[1] == ChaseAnalyser.RED:
                chaser_right = ChaseAnalyser.ChaserRight
            elif right_answer == ChaseAnalyser.CRight and seq[-2] == ChaseAnalyser.RED:
                chaser_right = ChaseAnalyser.ChaserRight
            else:
                chaser_right = ChaseAnalyser.ChaserWrong
            final.append(chaser_right)

        print(final)


class NumberAnalyser:

    # boilerplate code to pre-process image
    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    # thresholding
    def thresholding(self, image):
        gray = self.get_grayscale(image)
        (T, threshInv) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # visualize only the masked regions in the image
        masked = cv2.bitwise_not(gray, gray, mask=threshInv)
        ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
        ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
        ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
        return thresh4

    # dilation
    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    # erosion
    def erode(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    # opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    def canny(self, image):
        return cv2.Canny(image, 100, 200)

    # skew correction
    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated

    # template matching
    def match_template(self, image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    def numbers(self, img_path):

        reader = cv2.imread(img_path)
        # reader = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        pt.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/4.1.3/bin/tesseract'

        gray = self.get_grayscale(reader)
        thresh = self.thresholding(reader)
        opening = self.opening(reader)
        canny = self.canny(reader)
        noiseless = self.remove_noise(reader)

        # cv2.imshow('canny', canny)
        # cv2.waitKey(0)
        # cv2.imshow('gray', gray)
        # cv2.waitKey(0)
        # cv2.imshow('threshold', thresh)
        # cv2.waitKey(0)
        # cv2.imshow('opening', opening)
        # cv2.waitKey(0)
        # cv2.imshow('noise removal', noiseless)
        # cv2.waitKey(0)
        # cv2.imshow('og', reader)
        # cv2.waitKey(0)

        print('yes')
        print(pt.image_to_string(reader, config='--psm 11, -c tessedit_char_whitelist=$,0123456789'))

    def get_choice(self, img):
        x = ChaseAnalyser
        check = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        mid1 = check[900][1120]
        # print(mid1)
        mid2 = check[900][1775]
        top1 = check[900][1775]
        top2 = check[900][1775]
        low1 = check[900][1120]
        low2 = check[900][1775]
        if x.colour_checker(self, mid1) == ChaseAnalyser.GREEN or x.colour_checker(self, mid2) == ChaseAnalyser.GREEN:
            return (ChaseAnalyser.MidChoice)
        elif x.colour_checker(self, top1) == ChaseAnalyser.GREEN or x.colour_checker(self, top2) == ChaseAnalyser.GREEN:
            return ChaseAnalyser.TopChoice
        elif x.colour_checker(self, low1) == ChaseAnalyser.GREEN or x.colour_checker(self, low2) == ChaseAnalyser.GREEN:
            return ChaseAnalyser.LowChoice
        else:
            print('OH NO')


class TestClass:

    def test(self, test_file):
        chase_analyser = ChaseAnalyser()
        success = 0
        fail = 0
        with open(test_file, 'r') as read_obj:
            # Create a csv.reader object from the input file object
            csv_reader = reader(read_obj)
            next(csv_reader, None)
            for row in csv_reader:
                expected = row[3]
                got = str(chase_analyser.logic(row[2]))
                if expected == got:
                    print('success')
                    success += 1
                else:
                    print('fail ' + row[2])
                    print('expected ' + str(expected))
                    print('but got: ' + str(got))
                    fail += 1
            print(success, fail)


if __name__ == "__main__":
    x = ChaseAnalyser()
    x.get_frames('S5E115.mp4')
