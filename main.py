
from singlemotiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# инициализация output frame and a lock используемую для обеспечения безопасности потока обмена выходных кадров(полезно для просмотра потока различными браузерами)
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# инициализация flask object
app = Flask(__name__)

# initialize the video stream and позволяет датчикам камеры разогреться(приготовиться)
vs = VideoStream('http://camera.butovo.com/axis-cgi/mjpg/video.cgi?showlength=1').start()
time.sleep(2.0)

#  декоратор route() используется для привязки функции к URL
@app.route("/")
def index():
    # Для визуализации шаблона вы можете использовать метод render_template().
    # Всё, что вам необходимо - это указать имя шаблона, а также переменные в виде именованных аргументов,
    # которые вы хотите передать движку обработки шаблонов:
    return render_template("index.html")


def detect_motion(frameCount):
    # захватить глобальные ссылки на видеопоток, выходной кадр и блокированных переменных
    global vs, outputFrame, lock

    # initialize the motion detector and the total number of frames
    # read thus far
    # инициализация детектора движения и общее число кадров
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    # Цикл по кадрам видео потока

    while True:
        # прочитать следующий кадр из видео потока, изменить его размер,
        # преобразовать в grayscale и размываем его
        frame = vs.read()
        frame = imutils.resize(frame, width=800, height = 900)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        #возьмите текущую временную метку и нарисуйте ее на кадре
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # если общее количество кадров достигло достаточного количества для построения разумной фоновой модели,
        # то продолжайте обработку кадра
        if total > frameCount:
            # определяем движение на кадре
            motion = md.detect(gray)

            # проверяем было ли найдено движение на кадре
            if motion is not None:
                # расспаковываем кортеж и рисуем ограничительную рамку территории движения на выходном кадре
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                              (0, 0, 255), 2)

        # Обновить фоновую модель и увеличить количество считанных кадров на данныйц момент
        md.update(gray)
        total += 1

        # получить блокировку, установить выходной кадр и освободить блокиовку

        with lock:
            outputFrame = frame.copy()


def generate():
    # захватить глобальные ссылки на видеопоток, выходной кадр и блокированных переменных
    global outputFrame, lock

    # Цикл по всем выходным кадрам
    while True:
        # Ждём пока не появится блокировка
        with lock:
            # проверяем есть ли следующий выходной кадр, иначе прерываем итерацию цикла
            if outputFrame is None:
                continue

            # форматируем кадр в jpg
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # убеждаемся что форматирование прошло успешно
            if not flag:
                continue

        # вывести выходной кадр в байтовом формате
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    #
    # вернуть сгенерированный ответ вместе с конкретным типом медиа (тип mime)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


#
# проверить, является ли это основным потоком выполнения
if __name__ == '__main__':
    # запустить поток, который будет выполнять обнаружение движения
    t = threading.Thread(target=detect_motion, args=(
        32,))
    t.daemon = True
    t.start()

    # запускаем flask app
    app.run(host="127.0.0.1", port="8080", debug=True,
            threaded=True, use_reloader=False)

# освободить указатель видеопотока
vs.stop()
