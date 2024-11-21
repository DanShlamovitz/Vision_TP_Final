# Vision_TP_Final

En este trabajo práctico hacemos cosas piolas para en un futuro ser todavía más piolas


# Features 

FlickrId INT, -- Id of the image on Flickr
UserId TEXT, -- Id of the user on Flickr
URL TEXT, -- URL of the image on Flickr
Path TEXT,
DatePosted TEXT, -- Timestamp of the date of the image post on Flickr
DateTaken, -- Timestamp of the claimed date of the image creation
DateCrawl TEXT -- Timestamp of the crawling time
Camera TEXT, -- Camera model (if available)
Size INT, -- Total number of pixel of the original image (if available)
Title TEXT, -- Title of the post
Description TEXT, -- Description of the post
NumSets INT, -- Number of albums the photo is shared in
NumGroups INT, -- Number of groups the photo is shared in
AvgGroupsMemb REAL, -- Avg number of members of the groups in which the photo is shared
AvgGroupPhotos REAL, -- Avg number of photos of the groups in which the photo is shared
Tags TEXT, -- Social tags of the post
Latitude TEXT, -- (if available)
Longitude TEXT, -- (if available)
Country TEXT, -- (if available)
UserId TEXT, -- Id of the user on Flickr
Username TEXT,
Ispro INT, -- If the user's account is registered as professional
HasStats INT,
Contacts INT, -- Number of contacts of the user on Flickr
PhotoCount INT, -- Number of photos of the user
MeanViews REAL, -- Mean number of views of the user's photos
GroupsCount INT, -- Number of groups the user is enrolled in
GroupsAvgMembers REAL, -- Avg number of members of the groups in which the the user is enrolled
GroupsAvgPictures REAL -- Avg number of photos of the groups in which the the user is enrolled

