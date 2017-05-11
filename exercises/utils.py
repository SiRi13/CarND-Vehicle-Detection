import glob

def get_images():
    non_cars_images = glob.glob('../test_images/exercises/non-vehicles/*/*.png')
    cars_images = glob.glob('../test_images/exercises/vehicles/*/*.png')
    cars = []
    notcars = []

    for image_path in non_cars_images:
        notcars.append(image_path)

    for image_path in cars_images:
        cars.append(image_path)

    return cars, notcars
