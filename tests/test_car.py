# Copyright (C) 2024 Robert Bosch GmbH.
# The reproduction, distribution and utilization of this file as
# well as the communication of its contents to others without
# express authorization is prohibited. Offenders will be held
# liable for the payment of damages. All rights reserved in the
# event of the grant of a patent, utility model or design.

"""This module implements test for the Car class of the car module."""

import pytest

from mypackage.car import Car, main


@pytest.fixture
def car() -> Car:
    """
    Create a fixture that returns a new instance of the Car class.

    Returns:
        Car: A new instance of the Car class.

    """
    return Car("Toyota", "Camry", 2022)


def test_car_initialization(car: Car) -> None:
    """
    Test the initialization of the Car class.

    Args:
        car: An instance of the Car class.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # GIVEN a car of model 'Camry' from 'Toyota' of the year '2022'
    # WHEN I access the properties of the car
    # THEN they should be stored as follows and the mileage should be 0
    assert car.make == "Toyota"
    assert car.model == "Camry"
    assert car.year == 2022
    assert car.mileage == 0


def test_car_description(car: Car) -> None:
    """
    Test the get_description method of the Car class.

    Args:
        car: An instance of the Car class.

    Raises:
        AssertionError: If the description assertion fails.
    """
    # GIVEN a car of model 'Camry' from 'Toyota' of the year '2022'
    # WHEN I get the description of the car
    description = car.description

    # THEN is should be printed in this format
    assert description == "2022 Toyota Camry"


def test_car_drive(car: Car) -> None:
    """
    Test the get_mileage method of the Car class.

    Args:
        car: An instance of the Car class.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # GIVEN a car that has not been driven before
    car = Car("Toyota", "Yaris", 2023)

    # WHEN I drive it for 10 kilometers
    car.drive(10)

    # THEN the mileage should be 10 kilometers
    assert car.mileage == 10


def test_car_drive_multiple_times(car: Car) -> None:
    """
    Test the drive method of the Car class when driven multiple times.

    Args:
        car: An instance of the Car class.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # GIVEN a car that has not been driven before
    car = Car("Toyota", "Yaris", 2023)

    # WHEN I drive it multiple times
    car.drive(50)
    car.drive(30)

    # THEN the mileage should be the sum of those trips
    assert car.mileage == 80


def test_main(capsys) -> None:
    """
    Test the main function of the Car module.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # GIVEN nothing
    # WHEN I call the main function of the car module
    # THEN the result should be as follows.
    main()
    output = capsys.readouterr()
    assert output.out == "Car description: 2022 Toyota Camry\nCar mileage: 100 miles\n"
