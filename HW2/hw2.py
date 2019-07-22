import tkinter as tk
import numpy as np
import math
import cv2
from tkinter import filedialog 
from PIL import Image, ImageTk
import matplotlib.pyplot  as plt


class DIPGUI(tk.Frame):
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
		self.notation = tk.Label(self.bottomFrame, text = '是否保留其他 intensity', fg = 'black')
		self.mode = tk.IntVar()
		self.meth_box1 = tk.Radiobutton(self.bottomFrame, text = '不保留', variable = self.mode, value = 0, command = self.change_method, activebackground = 'red')
		self.meth_box2 = tk.Radiobutton(self.bottomFrame, text = '保留', variable = self.mode, value = 1, command = self.change_method, activebackground = 'red')
		self.notation.pack(anchor = tk.NW)
		self.meth_box1.pack(anchor = tk.NW)
		self.meth_box2.pack(anchor = tk.NW)

		# smooth, sharpen 選項
		self.smooth_btn = tk.Radiobutton(self.bottomFrame, text = 'smoothing', variable = self.mode, value = 2, command = self.change_method)
		self.sharpen_btn = tk.Radiobutton(self.bottomFrame, text = 'sharpen', variable = self.mode, value = 3, command = self.change_method)
		self.smooth_btn.pack(anchor = tk.NW)
		self.sharpen_btn.pack(anchor = tk.NW)

		# FFT, phase, spectrum 按鈕
		self.fft_btn = tk.Button(self.bottomFrame, text = 'show FFT2D result', width = 15, command = self.FFT_2D)
		self.fft_btn.pack(anchor = tk.NW)
		self.phase_btn = tk.Button(self.bottomFrame, text = 'phase image', width = 15, command = self.phase_image)
		self.phase_btn.pack(anchor = tk.NW)
		self.spectrum_btn = tk.Button(self.bottomFrame, text = 'spectrum image', width = 15, command = self.amplitude_image)
		self.spectrum_btn.pack(anchor = tk.NW)

		# 展示 bit plane 按鈕 
		self.bit = tk.IntVar()
		self.bit0 = tk.Radiobutton(self.bottomFrame, text = 'bit 0', variable = self.bit, value = 0, activebackground = 'red')
		self.bit1 = tk.Radiobutton(self.bottomFrame, text = 'bit 1', variable = self.bit, value = 1, activebackground = 'red')
		self.bit2 = tk.Radiobutton(self.bottomFrame, text = 'bit 2', variable = self.bit, value = 2, activebackground = 'red')
		self.bit3 = tk.Radiobutton(self.bottomFrame, text = 'bit 3', variable = self.bit, value = 3, activebackground = 'red')
		self.bit4 = tk.Radiobutton(self.bottomFrame, text = 'bit 4', variable = self.bit, value = 4, activebackground = 'red')
		self.bit5 = tk.Radiobutton(self.bottomFrame, text = 'bit 5', variable = self.bit, value = 5, activebackground = 'red')
		self.bit6 = tk.Radiobutton(self.bottomFrame, text = 'bit 6', variable = self.bit, value = 6, activebackground = 'red')
		self.bit7 = tk.Radiobutton(self.bottomFrame, text = 'bit 7', variable = self.bit, value = 7, activebackground = 'red')
		self.show_bit_plane_btn = tk.Button(self.bottomFrame, text = 'show bit plane', width = 15, command = self.show_bit_plane_img)
		self.bit0.pack(anchor = tk.E)
		self.bit1.pack(anchor = tk.E)
		self.bit2.pack(anchor = tk.E)
		self.bit3.pack(anchor = tk.E)
		self.bit4.pack(anchor = tk.E)
		self.bit5.pack(anchor = tk.E)
		self.bit6.pack(anchor = tk.E)
		self.bit7.pack(anchor = tk.E)
		self.show_bit_plane_btn.pack(anchor = tk.E)

		

		# 調整對比 slicing 區間卷轴
		self.upper_bound = tk.Scale(self.bottomFrame, fg = 'black', font = ('Arial', 12), label = '調整 slicing 上界' , from_ = 0, to = 255, orient = 'horizontal', length = 500, tickinterval = 25, resolution = 1)
		self.upper_bound.pack(side = 'top')
		self.upper_bound.pack()
		self.upper_bound.set(255)

		self.lower_bound = tk.Scale(self.bottomFrame, fg = 'black', font = ('Arial', 12), label = '調整 slicing 下界' , from_ = 0, to = 255, orient = 'horizontal', length = 500, tickinterval = 25, resolution = 1)
		self.lower_bound.pack(side = 'top')
		self.lower_bound.pack()
		self.lower_bound.set(0)


	def change_method(self):
		'''
		reset all the enhancement parameters and images as any of the methods is clicked
		'''
		mode = self.mode.get()
		if mode == 0 or mode == 1:   # grey level slicing 模式
			self.upper_bound.config(command = self.show_grey_level_slicing)
			self.lower_bound.config(command = self.show_grey_level_slicing)
			self.upper_bound.config(from_ = 0, to = 255, tickinterval = 25, resolution = 1)
			self.lower_bound.config(from_ = 0, to = 255, tickinterval = 25, resolution = 1)
			self.upper_bound.set(255)
			self.lower_bound.set(0)

		elif mode == 2:              # smooth 模式
			self.upper_bound.config(command = self.show_smooth)
			self.upper_bound.config(label = '選擇 spatial kernel 的邊長', from_ = 3, to = 13, tickinterval = 2, resolution = 1)
			self.upper_bound.set(3)

		elif mode == 3:              # sharpen 模式
			self.upper_bound.config(command = self.show_sharpen)
			self.upper_bound.config(label = '選擇 constant c 值(g(x,y = f(x,y) - c * 2nd derv of f))', from_ = 0, to = 10, tickinterval = 2, resolution = 0.5)
			self.upper_bound.set(1)
			arr = np.array(self.image)
			kernel = np.ones((7, 7))
			kernel /= (7**2)
			self.arr_smooth = self.conv(arr, kernel, 7)   # 做 sharpen 前要先將雜訊給過濾掉

		self.image_display = np.array(self.image)    # 还原图片
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

		if self.image.size[1] * self.image.size[0] > 300 * 300:    # 限制圖片最大顯示 500 x 500
			arr_cut = np.array(self.image)
			arr_cut = arr_cut[0 : 300, 0 : 300]
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

		self.meth_box1.select()
		self.bit0.select()
		self.change_method()


	def save_file(self):
		'''
		使用菜單讓使用者能儲存處理後的圖片
		'''
		ftypes = [('JPEG files', '*.jpg'), ('TIF files', '*.tif'), ('All files', '*')]	     # 設定 files 顯示種類
		new_fname = filedialog.asksaveasfilename(initialdir = "new_img.jpg", title = "Save file", filetypes = ftypes)
		cv2.imwrite(new_fname, np.array(self.image_display))


	def check_boundary(self):
		'''
		用來調整圖片顯示的大小, 預設為 350x350
		'''
		if self.image_display.shape[0] * self.image_display.shape[1] > 300 * 300:
			arr_cut = self.image_display.copy()
			arr_cut = Image.fromarray(arr_cut[0 : 300, 0 : 300])
			self.image_display = Image.fromarray(self.image_display)
			self.render_display = ImageTk.PhotoImage(arr_cut)
			self.img_box_adj.config(image = self.render_display)

		else:
			self.image_display = Image.fromarray(self.image_display)
			self.render_display = ImageTk.PhotoImage(self.image_display)
			self.img_box_adj.config(image = self.render_display)


	def grey_level_slicing(self, mode = 0):
		'''
		利用 mask 來實現 slicing
		'''
		if self.upper_bound.get() < self.lower_bound.get():
			self.image_display = np.array

		if mode == 0:      # 不保留其他的 intensity
			result = np.array(self.image)
			mask = np.logical_and(result >= self.lower_bound.get(), result <= self.upper_bound.get())
			result = np.zeros(result.shape)
			result[mask] = 125
			return result

		elif mode == 1:    # 保留其他 intensity
			result = np.array(self.image)
			mask = np.logical_and(result >= self.lower_bound.get(), result <= self.upper_bound.get())
			result[mask] = 255
			return result


	def show_grey_level_slicing(self, redun):
		self.image_display = self.grey_level_slicing(mode = self.mode.get())
		self.check_boundary()
	

	def bit_plane_img(self, bit):
		'''
		利用 mask 找出符合範圍的 intensity 並留下, 其餘都設成 0
		'''
		result = np.array(self.image)
		mask = np.logical_or(result < 2**(bit), result > 2**(bit + 1))
		result[mask] = 0
		return result


	def show_bit_plane_img(self):
		self.image_display = self.bit_plane_img(self.bit.get())
		self.check_boundary()
	

	def conv(self, arr, kernel, kernel_width = 3):
		'''
		實現 spatial convolution
		'''
		# zero padding
		arr_pad = np.pad(arr, (math.ceil(kernel_width / 2), math.ceil(kernel_width / 2)), mode = 'edge')
		result = np.zeros(arr.shape)

		for h in range(arr.shape[0]):
			for w in range(arr.shape[1]):
				vert_start = h
				vert_end = h + kernel_width
				horiz_start = w
				horiz_end = w + kernel_width

				result[h, w] = np.sum(np.multiply(arr_pad[vert_start : vert_end, horiz_start : horiz_end], kernel))

		return result


	def show_smooth(self, redun):
		kernel_width = self.upper_bound.get()
		arr = np.array(self.image)
		kernel = np.ones((kernel_width, kernel_width))     # 產生相對應大小的 box filter(kernel)
		kernel /= (kernel_width**2)                        # 標準化
		self.image_display = self.conv(arr, kernel, kernel_width)
		self.check_boundary()


	def show_sharpen(self, redun):   # 
		c = self.upper_bound.get()

		if c == 0:                   #  c = 0 代表不需要銳利化
			self.change_method()

		else:
			kernel_width = 3       
			arr = self.arr_smooth.copy()
			kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])     # 利用課本的 Laplacian filter 來實現
			self.image_display = arr - c * self.conv(arr, kernel, kernel_width)  # c 值越大代表越銳利
			self.check_boundary()


	def bit_reverse(self, arr):
		'''
		用來反轉一維陣列的順序 -> 方便 fft 進行運算
		'''
		n = len(arr)
		result = np.zeros(n, dtype = complex)
		bit_size = math.ceil(np.log2(n))

		for i in range(n):    # 針對每個 element 做對調
			bits = bin(i)
			result[int(bits[-1:1:-1] + (bit_size - len(bits[-1:1:-1])) * '0', 2)] = arr[i]

		return result


	def FFT(self, arr):
		'''
		1D FFT 轉換
		'''
		fft_m = self.bit_reverse(arr)
		fft_m = np.array(fft_m, dtype = complex)     # 

		# M = 2 ^ p = 2K 
		for p in range(1, int(np.log2(len(arr))) + 1):   # 共需要跑 log2(N) 次 (N -> 不斷分解成兩半進行運算)
			M = 2 ** p
			WM = np.exp(complex(0, -2 * math.pi / M))    # e^(-j2 * pi / M)

			for i in range(0, len(arr), M):              # 每 M 個 samples 進行一次 FFT
				w = 1
				for j in range(i, i + int(M / 2)):           # 只進行前半 (M / 2) 個 samples 就夠了, 剩下一半因對稱性查值就好
					e = fft_m[j]
					o = fft_m[j + int(M / 2)] * w
					fft_m[j] = e + o
					fft_m[j + int(M / 2)] = e - o
					w = w * WM

		return fft_m


	def FFT_2D(self, abs = True):
		'''
		2D FFT 實現
		'''
		fft_m_2D = np.array(self.image, dtype = complex)

		for i in range(fft_m_2D.shape[0]):               # 先做列轉換(1D)
			fft_m_2D[i, :] = self.FFT(fft_m_2D[i, :])


		for i in range(fft_m_2D.shape[1]):               # 在做行轉換(1D)
			fft_m_2D[:, i] = self.FFT(fft_m_2D[:, i])


		if abs == True:              # 是否顯示結果並回傳取絕對值後的 fft
			result = np.fft.fftshift(fft_m_2D)
			result = np.log(np.abs(result) + 1)
			#print(result)
			plt.imshow(result, 'gray')
			plt.show()
			return result

		return fft_m_2D        # 回傳未 shift 並 沒有取絕對值的 fft


	def phase_image(self):
		result = self.FFT_2D(abs = None)
		mapping = np.vectorize(lambda ele : np.exp(complex(0, np.angle(ele))))    # 進行 exp(angle) 運算
		result = mapping(result)
		result = np.fft.ifft2(result)         # 得到相角後做 inverse FFT

		result = np.real(result)    # 只拿 real part 來觀察
		plt.imshow(result, 'gray')
		plt.show()


	def amplitude_image(self):
		result = self.FFT_2D(abs = None)
		mapping = np.vectorize(lambda ele : (ele.real**2 + ele.imag**2)**0.5)    # 計算絕對值(amplitude)
		result = mapping(result)
		result = np.fft.fftshift(result)
		result = np.real(result)   # 只拿 real part 來觀察

		plt.imshow(result, 'gray')
		plt.show()



# 主函式
if __name__ == '__main__':
	window = tk.Tk()
	window.title('Hw1')
	window.resizable(True, True)
	gui = DIPGUI(master = window)
	gui.mainloop()

