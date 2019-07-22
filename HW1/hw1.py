import tkinter as tk
import numpy as np
import math
import cv2
from tkinter import filedialog 
from PIL import Image, ImageTk
import matplotlib.pyplot  as plt


class DIPGUI(tk.Frame):
	a_linear = 1
	b_linear = 0
	a_exp = 1
	b_exp = 0
	a_log = 1
	b_log = 0

	def __init__(self, master = None):
		super().__init__(master)
		self.master = master
		self.pack()
		self.create_widgets()
		self.file_name = ''
		self.image = None

	def create_widgets(self):
		# 定義菜單
		self.menubar = tk.Menu(self.master)           # main menu
		self.filemenu = tk.Menu(self.menubar, tearoff = 0)
		self.filemenu.add_command(label = 'Open File', command = self.open_file)
		self.filemenu.add_command(label = 'Save File', command = self.save_file)
		self.filemenu.add_separator()
		self.filemenu.add_command(label = 'Exit', command = self.master.quit)
		self.menubar.add_cascade(label = "File", menu = self.filemenu)
		self.master.config(menu = self.menubar)

		# 定義兩個主 Frames
		self.topFrame = tk.Frame(self, width = 1000, height = 450, bg = 'red')
		self.bottomFrame = tk.Frame(self, width = 1000, height = 500)
		
		self.topFrame.pack(side = 'top')
		self.bottomFrame.pack(side = 'bottom')

		# 圖片顯示區塊
		self.img_box_ori = tk.Label(self.topFrame, image = None)
		self.img_box_adj = tk.Label(self.topFrame, image = None)

		# methods checkbox (用來選擇哪種方法來處理圖片)
		self.notation = tk.Label(self.bottomFrame, text = 'interpolation methods', fg = 'black')
		self.interpolation = tk.StringVar()
		self.meth_box1 = tk.Radiobutton(self.bottomFrame, text = 'linear', variable = self.interpolation, value = 'linear', command = self.change_method, activebackground = 'red')
		self.meth_box2 = tk.Radiobutton(self.bottomFrame, text = 'exp', variable = self.interpolation, value = 'exp', command = self.change_method, activebackground = 'red')
		self.meth_box3 = tk.Radiobutton(self.bottomFrame, text = 'log', variable = self.interpolation, value = 'log', command = self.change_method, activebackground = 'red')
		self.notation.pack(anchor = tk.NW)
		self.meth_box1.pack(anchor = tk.NW)
		self.meth_box2.pack(anchor = tk.NW)
		self.meth_box3.pack(anchor = tk.NW)

		# 展示 histogram 按鈕 
		self.hist_btn = tk.Button(self.bottomFrame, text = 'show histogram', width = 15, command = self.show_histogram)
		self.hist_btn.pack(anchor = tk.E)
		self.equal_btn = tk.Button(self.bottomFrame, text = 'equalization', width = 15, command = self.equalization)
		self.equal_btn.pack(anchor = tk.E)


		# 調整對比 scale 卷轴
		self.constrast_scale_a = tk.Scale(self.bottomFrame, fg = 'black', font = ('Arial', 12), label = 'a 值' , from_ = 0.0, to = 5.0, orient = 'horizontal', length = 750, showvalue = 1, tickinterval = 0.5, resolution = 0.01)
		self.constrast_scale_a.pack(side = 'top')
		self.constrast_scale_a.pack()
		self.constrast_scale_a.set(1.00)

		self.constrast_scale_b = tk.Scale(self.bottomFrame, fg = 'black', font = ('Arial', 12), label = 'b 值' , from_ = -10.0, to = 10.0, orient = 'horizontal', length = 750, showvalue = 1, tickinterval = 1, resolution = 0.5)
		self.constrast_scale_b.pack(side = 'top')
		self.constrast_scale_b.pack()
		self.constrast_scale_b.set(0.0)

		# 缩放 scale 卷轴
		self.zoom_scale = tk.Scale(self.bottomFrame, bg = 'yellow', font = ('Arial', 12), label = 'zoom (100%)', from_ = 0.25, to = 3.00, orient = 'horizontal', length = 400, showvalue = 1, tickinterval = 0.250, resolution = 0.25)
		self.zoom_scale.pack(side = 'bottom')
		self.zoom_scale.set(1.00)

			
	def change_method(self):
		'''
		reset all the enhancement parameters and images as any of the methods is clicked
		'''
		self.reset_scale(self.interpolation.get())
		self.image_zoom = np.array(self.image.copy())
		self.image_display = self.adj_constrast(a = self.constrast_scale_a.get(), b = self.constrast_scale_b.get(), mode = self.interpolation.get())  # 还原图片
		self.check_boundary()


	def open_file(self):
		'''
		使用菜單讓使用者能選取要處理的圖片
		'''
		ftypes = [('JPEG files', '*.jpg'), ('TIF files', '*.tif'), ('All files', '*')]	   # 設定 files 顯示種類
		fdlg = tk.filedialog.Open(self.master, filetypes = ftypes)
		file_name = fdlg.show()
		file_name = file_name.split('/')[-1]

		self.image = Image.open(file_name)                  # 原圖片, 用來做處理的依據
		self.image_display = self.image.copy()              # 用來展示的圖片
		self.image_zoom = np.array(self.image.copy())                 # 用來檢查原圖是否已縮放過了


		if self.image.size[1] * self.image.size[0] > 500 * 500:    # 限制圖片最大顯示 500 x 500
			arr_cut = np.array(self.image)
			arr_cut = arr_cut[0 : 500, 0 : 500]
			self.render_orig = ImageTk.PhotoImage(Image.fromarray(arr_cut))
			self.render_display = ImageTk.PhotoImage(Image.fromarray(arr_cut))

		else:
			self.render_orig = ImageTk.PhotoImage(self.image)
			self.render_display = ImageTk.PhotoImage(self.image_display)	
	
		# 預設個調整紐的數值
		self.img_box_ori.configure(image = self.render_orig)
		self.img_box_ori.pack(side = 'left')

		self.img_box_adj.configure(image = self.render_display)
		self.img_box_adj.pack(side = 'right')

		self.zoom_scale.config(command = self.show_zoom_result)
		self.constrast_scale_a.config(command = self.show_adj_result)
		self.constrast_scale_b.config(command = self.show_adj_result)
		self.meth_box1.select()
		self.reset_scale('linear')



	def save_file(self):
		'''
		使用菜單讓使用者能儲存處理後的圖片
		'''
		ftypes = [('JPEG files', '*.jpg'), ('TIF files', '*.tif'), ('All files', '*')]	     # 設定 files 顯示種類
		new_fname = filedialog.asksaveasfilename(initialdir = "new_img.jpg", title = "Save file", filetypes = ftypes)
		cv2.imwrite(new_fname, np.array(self.image_display))
		#self.image_display.save(new_fname, 'TIFF')


	def reset_scale(self, mode = 'linear'):
		'''
		重新設定各模式的預設的對比調整值
		'''
		if mode == 'linear':
			self.constrast_scale_a.config(from_ = 0.0, to = 20.0, length = 750, showvalue = 1, tickinterval = 10, resolution = 0.05)
			self.constrast_scale_b.config(from_ = -20.0, to = 100.0, length = 750, showvalue = 1, tickinterval = 10, resolution = 1)
			self.constrast_scale_a.set(1.00)
			self.constrast_scale_b.set(0.0)

		elif mode == 'exp':
			self.constrast_scale_a.config(from_ = -0.02, to = 0.05, length = 750, showvalue = 1, tickinterval = 0.005, resolution = 0.001)
			self.constrast_scale_b.config(from_ = -5.0, to = 5.0, length = 750, showvalue = 1, tickinterval = 0.05, resolution = 0.025)
			self.constrast_scale_a.set(0.001)
			self.constrast_scale_b.set(2.5)

		else:
			self.constrast_scale_a.config(label = 'a 值 (以 10 為 base 的指數)', from_ = 1, to = 50, length = 750, showvalue = 10, resolution = 1)
			self.constrast_scale_b.config(from_ = 255, to = 1000.0, length = 750, showvalue = 1, tickinterval = 100, resolution = 10)
			self.constrast_scale_a.set(10.0)
			self.constrast_scale_b.set(255)

		self.zoom_scale.set(1.0)


	# 缩放函数
	def zoom(self, percentage = 1.5):
		'''
		此函式主要用來修改圖片大小（respect to the original size）
		percentage : 原圖放大縮小的比例
		return : numpy array
		'''
		ratio = 1 / percentage           # x, y 軸的縮放比例

		img_arr = np.array(self.image)
		result = np.zeros((int(self.image.size[0] * percentage), int(self.image.size[1] * percentage)))   # 結果

		for i in range(result.shape[0]):
			x_projected = float(i * ratio)
			u = x_projected - math.floor(x_projected)                     # y' 距離 y 的比例
			x_projected = math.floor(x_projected)

			# 檢查是否超出原圖片範圍
			if x_projected >= img_arr.shape[0] - 1 :
				x_projected = img_arr.shape[0] - 2
				u = 1

			if x_projected < 0 :
				x_projected = 0
				u = 0
			
			for j in range(result.shape[1]):
				y_projected = float(j * ratio)
				v = y_projected - math.floor(y_projected)
				y_projected = math.floor(y_projected)
				
				# 檢查是否超出原圖片範圍
				if y_projected >= img_arr.shape[1] - 1 :
					y_projected = img_arr.shape[1] - 2
					v = 1

				if y_projected < 0 :
					y_projected = 0
					v = 0

				result[i][j] = (1 - u) * (1 - v) * img_arr[x_projected][y_projected] + (1 - u) * v * img_arr[x_projected][y_projected + 1]
				result[i][j] += u * (1 - v) * img_arr[x_projected + 1][y_projected] + u * v * img_arr[x_projected + 1][y_projected + 1]

		if result.shape[0] * result.shape[1] != self.image_zoom.shape[0] * self.image_zoom.shape[1]:
			self.image_zoom = result.copy()           # 儲存縮放縮放後的 array (尚未調過對比度)

		# 再回傳縮放結果前先根據之前的a, b值做 enhance
		return self.adj_constrast(a = self.constrast_scale_a.get(), b = self.constrast_scale_b.get(), mode = self.interpolation.get())   

	def show_zoom_result(self, percentage):
		'''
		顯示縮放後的結果
		'''
		self.image_display = self.zoom(percentage = float(percentage))
		self.check_boundary()


	def adj_constrast(self, a = 1, b = 0, mode = 'linear'):
		'''
		調整圖片的對比度 分成 3 個 methods : linear, exponential, logarithmatic
		'''
		img_to_adj = None
		if self.image.size[0] * self.image.size[1] != self.image_zoom.shape[0] * self.image_zoom.shape[1]:   # 檢查圖片之前是否已縮放過
			img_to_adj = self.image_zoom.copy()
		else:
			img_to_adj = self.image.copy()


		if mode == 'linear':
			result = a * np.array(img_to_adj) + b

			# 設定 255 為 saturation, 只要 intensity 超過就設成 255, 較方便觀察
			mask1 = result > 255
			result[mask1] = 255

			#if np.max(result) > 255:
			#	result = self.normalization(result, 0, 255)

			return result

		elif mode == 'exp':
			result = np.exp(a * np.array(img_to_adj) + b)

			# 若最大值超過 255, 將整個 array normalize 回 [0, 255] 區間
			if np.max(result) > 255:
				result = self.normalization(result, 0, 255)

			return result

		else:
			result = np.log(np.array(10**a, dtype = 'float64') * np.array(img_to_adj, dtype = 'float64') + b)

			#if np.max(result) > 255:
			#	result = self.normalization(result, 0, 255)

			return result


	def show_adj_result(self, redun):
		'''
		顯示圖片調整對比後的結果圖
		'''
		self.image_display = self.adj_constrast(a = self.constrast_scale_a.get(), b = self.constrast_scale_b.get(), mode = self.interpolation.get())
		self.check_boundary()


	def check_boundary(self):
		'''
		用來調整圖片顯示的大小, 預設為 500x500
		'''
		if self.image_display.shape[0] * self.image_display.shape[1] > 500 * 500:
			arr_cut = self.image_display.copy()
			arr_cut = Image.fromarray(arr_cut[0 : 500, 0 : 500])
			self.image_display = Image.fromarray(self.image_display)
			self.render_display = ImageTk.PhotoImage(arr_cut)
			self.img_box_adj.config(image = self.render_display)
		else:
			self.image_display = Image.fromarray(self.image_display)
			self.render_display = ImageTk.PhotoImage(self.image_display)
			self.img_box_adj.config(image = self.render_display)


	def normalization(self, arr, a, b):
		'''
		將圖片重新 map 到 intensity 介於 [a, b] 間
		'''
		arr = (arr - np.min(arr)) * (b - a) / (np.max(arr) - np.min(arr)) + a
		return arr

	
	def histogram(self):
		'''
		統計目前圖片(self.image_display)的 histogram
		'''
		stats = [0] * 256         
		arr = np.array(self.image_display)

		rows = arr.shape[0] - 1
		cols = arr.shape[1] - 1

		for i in range(rows):
			for j in range(cols):
				if float(arr[i, j]) == float("-inf"):          # 預防指數運算後會有無限大或負無限大的情況
					stats[0] += 1 / ((rows + 1) * (cols + 1))
				elif float(arr[i, j]) == float("inf"):
					stats[255] += 1 / ((rows + 1) * (cols + 1))
				else:
					stats[round(float(arr[i, j]))] += 1 / ((rows + 1) * (cols + 1))

		return stats


	def show_histogram(self):
		'''
		畫出目前圖片(self.image_display)的 histogram
		'''
		stats = self.histogram()
		plt.plot(stats)
		plt.ylabel('probability')
		plt.xlabel('intensity')
		plt.show()	


	def equalization(self):
		'''
		根據課本提供的方式實現 histogram equalization, 並同時顯示處理後的結果
		'''
		r = self.histogram()
		arr = np.array(self.image_display)
		s = [0] * 256

		for i in range(256):   # 計算 s() 函數
			s[i] = 0
			for j in range(i):
				s[i] += 256 * r[j]

			s[i] = round(s[i])    # 四捨五入, 因 intensity 為整數
		
		mapping = np.vectorize(lambda a : s[int(a)])
		arr = mapping(arr)

	
		self.image_display = arr.astype(dtype = np.uint16)
		self.check_boundary()
		self.show_histogram()

		


# 主函式
if __name__ == '__main__':
	window = tk.Tk()
	window.title('Hw1')
	window.resizable(True, True)

	gui = DIPGUI(master = window)
	gui.mainloop()

