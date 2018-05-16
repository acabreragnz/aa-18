class Point:

    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.dimension = len(coordinates)

    def __repr__(self):
        return 'Coordinates: ' + str(self.coordinates) + \
               ' -> Dimension: ' + str(self.dimension)