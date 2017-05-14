import glob

def get_images(extension='jpeg', limit=None):
    non_cars_images = glob.glob('./test_images/non-vehicles/**/*.' + extension)
    cars_images = glob.glob('./test_images/vehicles/**/*.' + extension)
    cars = []
    notcars = []

    for idx, image_path in enumerate(non_cars_images):
        notcars.append(image_path)
        if limit is not None and idx >= limit-1:
            break

    for idx, image_path in enumerate(cars_images):
        cars.append(image_path)
        if limit is not None and idx >= limit-1:
            break

    return cars, notcars
