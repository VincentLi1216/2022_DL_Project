import random

def predict_age_gender_race():
    age = random.randint(1,116)
    gender = random.random()
    race = "FAKE_RACE"
    return age, gender, race

if __name__ == "__main__":
    print(predict_age_gender_race())
