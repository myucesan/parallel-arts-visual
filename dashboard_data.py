import pandas as pd
from pymongo import MongoClient
import re

client = MongoClient("localhost:27017")
db = client.deds

mapLocations = pd.DataFrame(
    list(db.reviews.find({}, {"Hotel_Name": 1, "Hotel_Address": 1, "Tags": 1, "lat": 1, "lng": 1})))
uniqueTags = pd.DataFrame(list(db.reviews.find({}, {"Hotel_Name": 1, "Tags": 1})))
uniqueTags = uniqueTags.drop_duplicates(["Hotel_Name"])["Tags"]
allTags = list()
for tags in uniqueTags:
    tags = re.sub(r"[\[\]']", "", tags).split(",")
    for tag in tags:
        tag = tag.strip()
        if tag not in allTags:
            allTags.append({'label': tag, 'value': tag})

mapLocations = mapLocations.drop_duplicates(["lat", "lng"])
mapLocations = mapLocations[mapLocations["lat"] != "NA"]
mapLocations = mapLocations[mapLocations["lng"] != "NA"]
mapLocations["lat"] = pd.to_numeric(mapLocations["lat"])
mapLocations["lng"] = pd.to_numeric(mapLocations["lng"])
mapLocations["Tags"] = mapLocations["Tags"].apply(str)
# mapLocations["Tags"] = mapLocations["Tags"].apply(
# #     lambda x: re.sub(r"[\[\]']", "", x).split(","))  # converts string to listie
# # mapLocations["Tags"] = mapLocations["Tags"].apply(
# #     lambda x: [i.strip() for i in x])  # takes every item in listie and removes the whitespace left and right
# mapLocations["Tags"] = mapLocations["Tags"].apply(lambda x: x.strip())

reviewerNationalityPipeline = [
    {
        u"$group": {
            u"_id": {
                u"Reviewer_Nationality": u"$Reviewer_Nationality"
            },
            u"COUNT(Reviewer_Nationality)": {
                u"$sum": 1
            }
        }
    },
    {
        u"$project": {
            u"Reviewer_Nationality": u"$_id.Reviewer_Nationality",
            u"COUNT(Reviewer_Nationality)": u"$COUNT(Reviewer_Nationality)",
            u"_id": 0
        }
    }
]
reviewerNationalityCountAggregate = pd.DataFrame(
    list(db.reviews.aggregate(reviewerNationalityPipeline, allowDiskUse=True)))
reviewerNationalityCountAggregate = reviewerNationalityCountAggregate[
    reviewerNationalityCountAggregate["COUNT(Reviewer_Nationality)"] > 5000]

from bson.code import Code

map = Code("function () {"
           "    emit(this.Hotel_Name, this.Additional_Number_Of_Scoring);"
           "}")

reduce = Code("function (key, values) {"
              "var total = 0;"
              "for (val in values) {"
              "total += parseInt(val);"
              "}"
	            "return total;"
              "}")
myresult = db.reviews.map_reduce(map, reduce, "additionalNumberOfScoringPerHotel")
additionalNumberDF = pd.DataFrame(
    list(db.additionalNumberOfScoringPerHotel.find({})))
