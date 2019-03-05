from imageai.Detection import ObjectDetection
import glob

# Set up detector
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()

# Load model
model_path = 'resnet50_coco_best_v2.0.1.h5'

detector.setModelPath(model_path)
detector.loadModel()

# Set custom object detection for dogs, cats, and birds
# For more options, see: https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection#---custom-object-detection
custom_objects = detector.CustomObjects(dog=True, cat=True, bird=True)

# Set up folder paths
input_images_path = 'dog-pictures/*'
output_images_path = 'results/'

# Iterate through folder of images
count = 0
for f in glob.glob(input_images_path):
    # Set output file path
    output_path = output_images_path + 'result_' + str(count) + '.jpg'

    # Detect
    print('Detecting...', output_path)
    detector.detectCustomObjectsFromImage(input_image=f, output_image_path=output_path, custom_objects=custom_objects, minimum_percentage_probability=40)

    count += 1

