import cv2

# Read the image
image = cv2.imread(r"D:\Downloads\saltandpeppernoise.jpg")
cv2.imshow('Original Image',image)

# Apply the 3x3 median filter on the image
processed_image = cv2.medianBlur(image, 3)
# display image
cv2.imshow('Median Filter Processing', processed_image)
