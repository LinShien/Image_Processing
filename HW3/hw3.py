import tkinter as tk
import numpy as np
import cv2
import math
import cmath
from tkinter import filedialog 
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


class DIP_GUI(tk.Frame):
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

		# smooth 、 sharpen 、homomorpihc filter 選擇
		self.notation_mode = tk.Label(self.bottomFrame, text = '請選擇 Scale 模式', fg = 'black')
		self.mode = tk.IntVar()
		self.meth_box1 = tk.Radiobutton(self.bottomFrame, text = 'homomorphic filter mode', variable = self.mode, value = 1, command = self.change_method, activebackground = 'red')
		self.meth_box2 = tk.Radiobutton(self.bottomFrame, text = 'RGB image smoothing mode', variable = self.mode, value = 2, command = self.change_method, activebackground = 'red')
		self.meth_box3 = tk.Radiobutton(self.bottomFrame, text = 'RGB image sharpening mode', variable = self.mode, value = 3, command = self.change_method, activebackground = 'red')
		self.notation_mode.pack(anchor = tk.W)
		self.meth_box1.pack(anchor = tk.W)
		self.meth_box2.pack(anchor = tk.W)
		self.meth_box3.pack(anchor = tk.W)

		# 其他功能的按鈕
		self.homo_filter_btn = tk.Button(self.bottomFrame, text = 'Homomorphic filter', width = 20, command = self.show_homorphic_filter)
		self.homo_filter_btn.pack(anchor = tk.W)
		self.smooth_btn = tk.Button(self.bottomFrame, text = 'smoothing', width = 15, command = self.show_smoothing)
		self.smooth_btn.pack(anchor = tk.W)
		self.sharpen_btn = tk.Button(self.bottomFrame, text = 'sharpen', width = 15, command = self.show_sharpening)
		self.sharpen_btn.pack(anchor = tk.W)
		self.rgb_btn = tk.Button(self.bottomFrame, text = '顯示 RGB components', width = 20, command = self.show_RGB)
		self.rgb_btn.pack(anchor = tk.W)
		self.rgb2hsi_btn = tk.Button(self.bottomFrame, text = '顯示 RGB to HSI', width = 20, command = self.show_HSI)
		self.rgb2hsi_btn.pack(anchor = tk.W)
		self.rgb_complement_btn = tk.Button(self.bottomFrame, text = 'RGB complement', width = 20, command = self.rgb_complement)
		self.rgb_complement_btn.pack(anchor = tk.W)
		self.segmentation_btn = tk.Button(self.bottomFrame, text = 'Segmentation', width = 20, command = self.segmentation)
		self.segmentation_btn.pack(anchor = tk.W)

		# rL, rH, c, D0 Scale wedgets
		self.rL = tk.Scale(self.bottomFrame, font = ('Arial', 12), label = 'rL', from_ = 0.1, to = 1.9, orient = 'horizontal', length = 200, showvalue = 1, tickinterval = 1, resolution = 0.1)
		self.rH = tk.Scale(self.bottomFrame, font = ('Arial', 12), label = 'rH', from_ = 2.0, to = 7.0, orient = 'horizontal', length = 500, showvalue = 1, tickinterval = 1, resolution = 0.1)
		self.c = tk.Scale(self.bottomFrame, font = ('Arial', 12), label = 'c', from_ = 0.1, to = 10, orient = 'horizontal', length = 500, showvalue = 1, tickinterval = 2, resolution = 0.1)
		self.D0 = tk.Scale(self.bottomFrame, font = ('Arial', 12), label = 'D0', from_ = 20, to = 1200, orient = 'horizontal', length = 1000, showvalue = 1, tickinterval = 100, resolution = 20)
		self.rL.pack(side = 'bottom')
		self.rH.pack(side = 'bottom')
		self.c.pack(side = 'bottom')
		self.D0.pack(side = 'bottom')

		self.rL.set(0.4)
		self.rH.set(3.0)
		self.c.set(5.0)
		self.D0.set(20)
		
	def reset_scale(self, mode = 1):
		'''
		重新設定各模式的預設的參數調整值
		'''
		if mode == 1:
			self.rL.config(label = 'rL', from_ = 0.1, to = 1.9, orient = 'horizontal', length = 200, showvalue = 1, tickinterval = 1, resolution = 0.1)
			self.rH.config(label = 'rH', from_ = 2.0, to = 7.0, orient = 'horizontal', length = 500, showvalue = 1, tickinterval = 1, resolution = 0.1)
			self.c.config(label = 'c', from_ = 0.1, to = 10, orient = 'horizontal', length = 500, showvalue = 1, tickinterval = 2, resolution = 0.1)
			self.D0.config(label = 'D0', from_ = 20, to = 1200, orient = 'horizontal', length = 1000, showvalue = 1, tickinterval = 100, resolution = 20)
			self.rL.set(0.4)
			self.rH.set(3.0)
			self.c.set(5.0)
			self.D0.set(20)

		elif mode == 2:
			self.D0.config(label = 'box filter 寬度', from_ = 3, to = 10, length = 750, showvalue = 1, tickinterval = 10, resolution = 1)
			self.D0.set(5)

		elif mode == 3:
			self.D0.config(label = '選擇常數 c 值', from_ = -4.0, to = 0.0, length = 750, showvalue = 1, tickinterval = 10, resolution = 0.1)
			self.D0.set(0.0)
		

	def change_method(self):
		'''
		reset all the enhancement parameters and images as any of the methods is clicked
		'''
		self.reset_scale(self.mode.get())
		self.image_display = np.array(self.image.copy())  # 还原图片
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


		if self.image.size[1] * self.image.size[0] > 350 * 350:    # 限制圖片最大顯示 500 x 500
			arr_cut = np.array(self.image)
			arr_cut = arr_cut[0 : 350, 0 : 350]
			self.render_orig = ImageTk.PhotoImage(Image.fromarray(arr_cut))
			self.render_display = ImageTk.PhotoImage(Image.fromarray(arr_cut))

		else:
			self.render_orig = ImageTk.PhotoImage(self.image)
			self.render_display = ImageTk.PhotoImage(self.image_display)	
		
		
		# 顯示圖片
		self.img_box_ori.configure(image = self.render_orig)
		self.img_box_ori.pack(side = 'left')

		self.img_box_adj.configure(image = self.render_display)
		self.img_box_adj.pack(side = 'right')

		# 預先勾選選項
		self.meth_box1.select()
		self.reset_scale()


	def save_file(self):
		'''
		使用菜單讓使用者能儲存處理後的圖片
		'''
		ftypes = [('JPEG files', '*.jpg'), ('TIF files', '*.tif'), ('All files', '*')]	     # 設定 files 顯示種類
		new_fname = filedialog.asksaveasfilename(initialdir = "new_img.jpg", title = "Save file", filetypes = ftypes)
		cv2.imwrite(new_fname, np.array(self.image_display))


	def check_boundary(self, rgb = True):
		'''
		用來調整圖片顯示的大小, 預設為 350 x 350
		'''
		img = np.array(self.image_display).astype('uint8')

		if rgb != True:          # 檢查是否通道排列順序為 BGR
			R = img[:, :, 2]
			G = img[:, :, 1]
			B = img[:, :, 0]
			img = np.stack((R, G, B), axis = 2)

		if self.image.size[1] * self.image.size[0] > 350 * 350:    # 限制圖片最大顯示 500 x 500
			arr_cut = img[0 : 350, 0 : 350]
			self.render_display = ImageTk.PhotoImage(Image.fromarray(arr_cut))
			self.img_box_adj.config(image = self.render_display)

		else:
			self.render_display = ImageTk.PhotoImage(img)
			self.img_box_adj.config(image = self.render_display)
			

	def homomorphic_filter(self):
		'''
		實現 (4-147)式 的 homomorphic filter 
		'''
		rL = self.rL.get()
		rH = self.rH.get()
		c = self.c.get()

		z_orig = np.array(self.image).astype('f8')
		z = z_orig + 0.01                             # + 0.01 防止對數取 0 
		D = np.zeros(z.shape)
		H = np.zeros(z.shape)
		D0 = self.D0.get()
		P = np.floor(z.shape[0] / 2)
		Q = np.floor(z.shape[1] / 2)

		z_log = np.log2(z)              # 先取對數，再做 DFT，再 shift 到中心
		Z_fft = np.fft.fft2(z_log)  
		Z_shift = np.fft.fftshift(Z_fft)             

		for u in range(D.shape[0]):
			for v in range(D.shape[1]):
				H[u, v] = (rH - rL) * (1 - np.exp(-(c * (((u - P)**2 + (v - Q)**2) / (D0**2))))) + rL  # 4-147 式的實現

		S = Z_shift * H                       	  		 # 結果相乘
		s = np.fft.ifft2(np.fft.ifftshift(S))   		# 先 shift 回原點，再做 IDFT
		s = np.real(s)                          		# 只取實部 
		g = np.exp(s) - 0.01                   			 # 取 exp 後再把 0.01 扣回來
		
		g = (g - np.min(g)) * 255 / (np.max(g) - np.min(g))  # normalization

		plt.subplot(1, 2, 1)
		plt.imshow(z_orig, 'gray')
		plt.title("original image")

		plt.subplot(1, 2, 2)
		plt.imshow(g, 'gray')
		plt.title("result of homomorphic filter")
		plt.show() 

		return g
	
	
	def show_homorphic_filter(self):
		self.image_display = self.homomorphic_filter()
		self.check_boundary()


	def show_RGB(self):
		'''
		先各自取出 R, G, B 三個 channels，在各自變成 24 bits 的圖片，
		最後把其他兩個不重要的 channels 都射成 0，達到只顯示 R，G，B 的效果
		'''
		img = np.array(self.image)
		R = img.copy()
		R[:, :, 1] = R[:, :, 2] = 0

		G = img.copy()
		G[:, :, 0] = G[:, :, 2] = 0

		B = img.copy()
		B[:, :, 0] = B[:, :, 1] = 0

		#cv2.imwrite("R.jpg", B)
		#cv2.imwrite("G.jpg", G)
		#cv2.imwrite("B.jpg", R)

		plt.subplot(2, 2, 1)
		plt.imshow(img)
		plt.title("original image")

		plt.subplot(2, 2, 2)
		plt.imshow(R)
		plt.title("Red component")

		plt.subplot(2, 2, 3)
		plt.imshow(G)
		plt.title("Green component")

		plt.subplot(2, 2, 4)
		plt.imshow(B)
		plt.title("Blue component")

		plt.show()


	def show_HSI(self):
		'''
		實現 RGB to HSI
		'''
		img_ori = np.array(self.image)
		img = img_ori.copy().astype(float)
		# RGB 都先各自取出來
		R = img[:, :, 0]
		G = img[:, :, 1]
		B = img[:, :, 2]

		# 計算角度
		den = ((R - G)**2 + (R - B) * (G - B))**0.5 + 0.0000000001       # 計算課本 6-17 式的分母部分, 並加上極小值，防止產生分母為 0 的情況
		theta = np.degrees(np.arccos((0.5 * ((R - G) + (R - B))) / den).astype('f8'))   # 計算 6-17 式，結果為弧度而非角度, 所以要用 np.degrees 換成角度

		H = np.zeros((img.shape[0], img.shape[1])).astype('f8')
		I = np.zeros((img.shape[0], img.shape[1])).astype('f8')
		S = np.zeros((img.shape[0], img.shape[1])).astype('f8')
		mmin = np.zeros((img.shape[0], img.shape[1])).astype('f8')

		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				# 處理課本 6-16 式, 即 H 的部分
				if den[i, j] == 0:
					H[i, j] = 0;
				elif B[i, j] <= G[i, j]:  # B <= G
					H[i, j] = theta[i, j]
				else:
					H[i, j] = 360 - theta[i, j]

				mmin[i, j] = min((R[i, j], G[i, j], B[i, j]))  # 計算 6-18 式中 min 的部分
				I[i, j] = (R[i, j] + G[i, j] + B[i, j]) / 3.0  # 計算 6-19 式, 即 I 的部分

		S = 1 - (1 / I) * mmin   # 計算 S 的部分

		#cv2.imwrite("hue.jpg", (H / 360) * 255)
		#cv2.imwrite("sat.jpg", S * 255)
		#cv2.imwrite("int.jpg", I * 255)

		plt.subplot(2, 2, 1)
		plt.imshow(img_ori)
		plt.title("original image")

		plt.subplot(2, 2, 2)
		plt.imshow((H / 360) * 255, 'gray')
		plt.title("Hue")

		plt.subplot(2, 2, 3)
		plt.imshow(S * 255, 'gray')
		plt.title("Saturation")

		plt.subplot(2, 2, 4)
		plt.imshow(I * 255, 'gray')
		plt.title("Intensity")
		plt.show()	
        
		return H, S, I


	def rgb_complement(self):
		'''
		在 RGB color space 下實現 color complement
		'''
		img = np.array(self.image)
		# 各 channel 都做負片效果
		R = 255 - img[:, :, 0]     
		G = 255 - img[:, :, 1]
		B = 255 - img[:, :, 2]

		self.image_display = np.stack((R, G, B), axis = 2)  # 最後在合成 24 bits 圖片
		self.check_boundary()


	def conv(self, arr_cut, kernel):
		'''
		convolution 實現
		'''
		result = np.sum(arr_cut * kernel)
		return result                         # result is a scalar here


	def smooth(self, kernel_width = 5):
		'''
		利用課本的 box filter kernels 來實現 smoothing
		'''
		img = np.array(self.image).astype('f8')
		img_pad = np.pad(img, ((int(kernel_width / 2), int(kernel_width / 2)), (int(kernel_width / 2), int(kernel_width / 2)), (0, 0)), mode = 'edge')
		n_H = int(img.shape[0])
		n_W = int(img.shape[1])
		result = np.zeros((n_H, n_W, 3))                    # 先創一個要輸出的矩陣, 初始化為 0
		kernel = np.ones((kernel_width, kernel_width))

		for h in range(n_H):
			for w in range(n_W):
				vertical_start = h 
				vertical_end = h + kernel_width
				horizontal_start = w
				horizontal_end = w + kernel_width

				# 對於 RGB 三個 channels 都要各自做 convolution
				arr_cut = img_pad[vertical_start : vertical_end, horizontal_start : horizontal_end, 0]
				result[h, w, 2] = int(self.conv(arr_cut, kernel) / (kernel_width ** 2))

				arr_cut = img_pad[vertical_start : vertical_end, horizontal_start : horizontal_end, 1]
				result[h, w, 1] = int(self.conv(arr_cut, kernel) / (kernel_width ** 2))

				arr_cut = img_pad[vertical_start : vertical_end, horizontal_start : horizontal_end, 2]
				result[h, w, 0] = int(self.conv(arr_cut, kernel) / (kernel_width ** 2))

		return result


	def show_smoothing(self):
		self.image_display = self.smooth(kernel_width = self.D0.get())
		self.check_boundary(rgb = False)


	def sharpen(self, c):
		img = np.array(self.image).astype('f8')
		img_pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode = 'constant')	

		n_H = int(img.shape[0])
		n_W = int(img.shape[1])
		result = np.zeros((n_H, n_W, 3))                    # 先創一個要輸出的矩陣, 初始化為 0
		kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

		for h in range(n_H):
			for w in range(n_W):
				vertical_start = h 
				vertical_end = h + 3
				horizontal_start = w
				horizontal_end = w + 3

				# 對於 RGB 三個 channels 都要各自做 convolution
				arr_cut = img_pad[vertical_start : vertical_end, horizontal_start : horizontal_end, 0]
				result[h, w, 2] = int(self.conv(arr_cut, kernel) / (3 ** 2))

				arr_cut = img_pad[vertical_start : vertical_end, horizontal_start : horizontal_end, 1]
				result[h, w, 1] = int(self.conv(arr_cut, kernel) / (3 ** 2))

				arr_cut = img_pad[vertical_start : vertical_end, horizontal_start : horizontal_end, 2]
				result[h, w, 0] = int(self.conv(arr_cut, kernel) / (3 ** 2))
		
		# 最後每個 channels 都做標準化
		img[:, :, 0] = (img[:, :, 0] - np.min(img[:, :, 0])) * 255 / (np.max(img[:, :, 0]) - np.min(img[:, :, 0]))
		img[:, :, 1] = (img[:, :, 1] - np.min(img[:, :, 1])) * 255 / (np.max(img[:, :, 1]) - np.min(img[:, :, 1]))
		img[:, :, 2] = (img[:, :, 2] - np.min(img[:, :, 2])) * 255 / (np.max(img[:, :, 2]) - np.min(img[:, :, 2]))
		img = np.stack((img[:, :, 2], img[:, :, 1], img[:, :, 0]), axis = 2)
	
		return result * c + img


	def show_sharpening(self):
		self.image_display = self.sharpen(c = self.D0.get())
		self.check_boundary(rgb = False)


	def segmentation(self):
		img = np.array(self.image)
		H, S, I = self.show_HSI()
		H_copy = H / 360.0 * 255.0    # 先標準化到 [0, 255]
		S_copy = S * 255.0
		S_neg = 255.0 - S_copy      # 取負片
    	
    	# Hue 上做 intensity slicing
		mask1 = np.logical_and(H_copy < 230.0, H_copy > 200.0)
		mask2 = H_copy >= 230.0
		mask3 = H_copy <= 200.0
		H_sliced = H_copy.copy()
		H_sliced[mask3] = 0
		H_sliced[mask2] = 0
		H_sliced[mask1] = 1
        
        # 再做一次 intensity slicing
		result = H_sliced * S_neg
		mask4 = result > 140.0
		mask5 = result <= 140.0
		result_sliced = result.copy()
		result_sliced[mask4] = 255.0
		result_sliced[mask5] = 0.0
        
		plt.subplot(2, 4, 1)
		plt.imshow(img)
		plt.title("original image")

		plt.subplot(2, 4, 2)
		plt.imshow(H_copy, 'gray')
		plt.title("Hue")

		plt.subplot(2, 4, 3)
		plt.imshow(S_copy, 'gray')
		plt.title("Sat")

		plt.subplot(2, 4, 4)
		plt.imshow(S_neg, 'gray')
		plt.title("Sat negative")

		plt.subplot(2, 4, 5)
		plt.imshow(H_sliced, 'gray')
		plt.title("Hue first time slicing")

		plt.subplot(2, 4, 6)
		plt.imshow(result, 'gray')
		plt.title("product of Hue_sliced and Sat_neg")

		plt.subplot(2, 4, 7)
		plt.imshow(result_sliced, 'gray')
		plt.title("segmentation of feathears")
		plt.show()	
             

# 主函式
if __name__ == '__main__':
	window = tk.Tk()
	window.title('Hw3')
	window.resizable(True, True)

	gui = DIP_GUI(master = window)
	gui.mainloop()
