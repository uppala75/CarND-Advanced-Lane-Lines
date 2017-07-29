import numpy as np
import cv2
class tracker():
	
	def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym = 1, My_xm = 1, Mysmooth_factor = 15):
		#list that stores all the past (left,right,center) set values used for smoothing the output
		self.recent_centers = []
		
		# The window pixel width of the center values, used to count pixels inside center windows to determine curve values
		self.window_width = Mywindow_width
		
		# The window pixel height of the center values, used to count pixels inside center windows to determine curve values
		# breaks the image into vertical levels
		self.window_height = Mywindow_height
		
		# The pixel distance in both directions to slide (left_window + right_window) template for searching
		self.margin = Mymargin
		
		self.ym_per_pix = My_ym # meters per pixel in vertical axis
		
		self.xm_per_pix = My_xm # meters per pixel in horizontal axis
		
		self.smooth_factor = Mysmooth_factor
		
	# Main tracking function for finding & storing the lane segment positions
	def find_window_centroids(self, warped):
		
			window_width = self.window_width
			window_height = self.window_height
			margin = self.margin
			
			window_centroids = [] # store the (left,right) window centroid positions per level
			window = np.ones(window_width) # create our window template that we will use for convolutions
			
			# first find the two starting positions for the left & right lane by using np.sum to get the vertical image slice
			# and then np.convolve the vertical image slice with the window template
			
			#sum quarter bottom of image to get slice, could use a different ratio
			l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
			l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
			r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
			r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
			
			# Add what we found for the 1st layer
			window_centroids.append((l_center,r_center))
			
			# Go thru each layer looking for max pixel locations
			for level in range(1,(int)(warped.shape[0]/window_height)):
				# convolve the window into the vertical slice of the image
				image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
				conv_signal = np.convolve(window, image_layer)
				# Find the best left centroid by using past left center as a reference
				# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
				offset = window_width/2
				l_min_index = int(max(l_center+offset-margin,0))
				l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
				l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
				# Find the best right centroid by using past right center as a reference
				r_min_index = int(max(r_center+offset-margin,0))
				r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
				r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
				# Add what we found for that layer
				window_centroids.append((l_center,r_center))

			self.recent_centers.append(window_centroids)
			# return averaged values of the line centers, helps to keep the markers from sliding too much
			return np.average(self.recent_centers[-self.smooth_factor:], axis=0)
		
		