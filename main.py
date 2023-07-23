import cv2
import time
import os
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO

#initiliaze GPIO
pinOUt = 22
GPIO.setup(pinOut, GPIO.OUT)


#Load bottle classifier

def output_converter(model_output):

    import numpy as np

    output = model_output

    # assume that 'output' is a numpy array of shape (n, 3)
    output_labels = ['0.5 L', '1 L', '1.5 L', '2 L']
    predictions = np.argmax(output, axis=1)
    predicted_labels = [output_labels[p] for p in predictions]

    return predicted_labels


while True:

  # Create a VideoCapture object
  cap = cv2.VideoCapture(0)

  # Capture an image
  ret, frame = cap.read()

  # Check if the capture was successful
  if ret:

      # Resize the image to 64x64
      rescaled_image = cv2.resize(frame, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

      # Convert the image to an array
      image_array = np.array(rescaled_image, dtype=np.uint8)

      # Display the image
      cv2.imshow("Image", rescaled_image)

      # Save the image
      save_directory = "/my/new/directory"
      cv2.imwrite(os.path.join(save_directory, "image.jpg"), rescaled_image)

      # Wait for a key press
      cv2.waitKey(0)

  # Release the VideoCapture object
  cap.release()
  
  model = tf.keras.models.load_model("/content/images/self_train.h5")
  prediction_label = output_converter(model.predict(image_array.reshape(1, 64, 64, 3)))
  
  if prediction_label = '0.5 L':
    label = 0
    GPIO.output(pinOut, GPIO.LOW)
    print('Bottle is 0.5 L.')
  else:
    label = 1
    GPIO.output(pinOut, GPIO.HIGH)
    print('Bottle is 1 L.')

  time.sleep(10)
