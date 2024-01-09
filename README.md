# Spaceship Titanic

The [task](https://www.kaggle.com/competitions/spaceship-titanic/overview) is
predict whether a passenger was transported to an alternate dimension during the
Spaceship Titanic's collision with the spacetime anomaly.

Project within the framework of the course Applied MLOps 2023 of the MSc program
of the National Research University Higher School of Economics "Machine Learning
and Data-Intensive Systems".

## Data description

- train data is consist of ~8700 personal records for about two-thirds of the
  passengers.
- test data is consist of remaining ~4300 personal records.

**The data has the following structure:**

- PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp
  where gggg indicates a group the passenger is travelling with and pp is their
  number within the group. People in a group are often family members, but not
  always.
- HomePlanet - The planet the passenger departed from, typically their planet of
  permanent residence.
- CryoSleep - Indicates whether the passenger elected to be put into suspended
  animation for the duration of the voyage. Passengers in cryosleep are confined
  to their cabins.
- Cabin - The cabin number where the passenger is staying. Takes the form
  deck/num/side, where side can be either P for Port or S for Starboard.
- Destination - The planet the passenger will be debarking to.
- Age - The age of the passenger.
- VIP - Whether the passenger has paid for special VIP service during the
  voyage.
- RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has
  billed at each of the Spaceship Titanic's many luxury amenities.
- Name - The first and last names of the passenger.
- Transported - Whether the passenger was transported to another dimension. This
  is the target.

## Initial configuration

Tested for Python=3.11

```
conda create --name <virtual_name>
pip install poetry
poetry config virtualenvs.create false
poetry install
```

## Experiments reproduction

```
mlflow ui
python train.py
python infer.py
```

## General scheme

Missing value preprocessing, one-hot-encoding for categorical features and
standardscaler for numeric features.

Simple MLP model over CrossEntropyLoss() with dropout and optimizer Adam.
