# Copyright (C) 2024 Robert Bosch GmbH.
# The reproduction, distribution and utilization of this file as
# well as the communication of its contents to others without
# express authorization is prohibited. Offenders will be held
# liable for the payment of damages. All rights reserved in the
# event of the grant of a patent, utility model or design.

"""This module provides a Car class for representing cars and demonstrating its usage.

Module: car

Classes:
    Car: A class representing a car.
"""

import logging


class Car:
    """A class representing a car.

    Attributes:
        make (str): The make of the car.
        model (str): The model of the car.
        year (int): The manufacturing year of the car.
        mileage (int): The current mileage of the car in miles.
        description (int): The description of the car in format "year make model".
    """

    def __init__(self, make: str, model: str, year: int) -> None:
        """Initialize an instance of a car.

        Args:
            make: The make of the car.
            model: The model of the car.
            year: The manufacturing year of the car.
        """
        self.__make = make
        self.__model = model
        self.__year = year
        self.__mileage = 0

        logger = logging.getLogger(__name__)
        logger.info("Car created: %s", self.description)

    def drive(self, distance: int) -> None:
        """Drive the car for a given distance.

        Args:
            distance: The distance to drive in miles.
        """
        self.__mileage += distance

    @property
    def make(self) -> str:
        """Return the make of the car."""
        return self.__make

    @property
    def model(self) -> str:
        """Return the model of the car."""
        return self.__model

    @property
    def year(self) -> int:
        """Return the year of the car."""
        return self.__year

    @property
    def mileage(self) -> int:
        """Return the current mileage of the car."""
        return self.__mileage

    @property
    def description(self) -> str:
        """Return the car description in the format "year make model"."""
        return f"{self.__year} {self.__make} {self.__model}"


def main() -> None:
    """Main function to demonstrate the usage of the Car class."""
    # Create a new instance of Car

    car = Car("Toyota", "Camry", 2022)

    # Drive the car for different distances
    car.drive(50)
    car.drive(30)
    car.drive(20)

    # Get the car's description and mileage
    description = car.description
    mileage = car.mileage

    # Print the car's information
    print(f"Car description: {description}")
    print(f"Car mileage: {mileage} miles")


if __name__ == '__main__':
    main()