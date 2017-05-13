import glob

def get_images(extension='jpeg'):
    non_cars_images = glob.glob('../test_images/non-vehicles/**/*.' + extension)
    cars_images = glob.glob('../test_images/vehicles/**/*.' + extension)
    cars = []
    notcars = []

    for image_path in non_cars_images:
        notcars.append(image_path)

    for image_path in cars_images:
        cars.append(image_path)

    return cars, notcars
