# import the necessary packages
import numpy as np
import imutils
import cv2

class SingleMotionDetector:
	def __init__(self, accumWeight=0.5):
		# store the accumulated weight factor
		# хранить накопленный весовой коэффициент
		self.accumWeight = accumWeight

		# initialize the background model
		# инициализация пред модели
		self.bg = None

	def update(self, image):
		# Если пред модель =  None, инициализируем ее
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return


		# Обновление пред модели
		# Функция вычисляет взвешенную сумму входного изображения image
		# и накопителя self.bg, так что self.bg становится скользящим средним для последовательности кадров
		# self.accumWeight - Вес входного изображения
		cv2.accumulateWeighted(image, self.bg, self.accumWeight)

	def detect(self, image, tVal=25):

		# вычислить абсолютную разницу между фоновой моделью и передаваемым изображением,
		# затем ищем пороговое значение для delta
		delta = cv2.absdiff(self.bg.astype("uint8"), image)
		thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]


		#  серия операций эрозии и делатации для удаления маленьких капель(шумов?)
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)


		# Поиск контуров в изображении после нахождения порога и инициализация минимума и максимума границ прямоугольника
		# для региона с движениями
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)

		# Если не найдено контуров -  None
		if len(cnts) == 0:
			return None

		# иначе цикл по контурам
		for c in cnts:

			# вычислить ограничивающую рамку контура и использовать его для обновления максимума и минимума
			# ограничивающей рамки контура
			(x, y, w, h) = cv2.boundingRect(c)
			(minX, minY) = (min(minX, x), min(minY, y))
			(maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

		# Возврат кортежа изображения с пороговым значением и с ограничивающимирамками
		return (thresh, (minX, minY, maxX, maxY))