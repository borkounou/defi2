# Import the necessary libraries
from bs4 import BeautifulSoup
import pandas as pd
import csv
from constants import PATH_XML_TEST, PATH_CSV_TEST

# Open and read the xml file with beautifulSoup class: The encoding of the file  is utf-8
with open(PATH_XML_TEST, 'r', encoding='utf-8') as fhandle:
    soup = BeautifulSoup(fhandle, 'xml')

if __name__=='__main__':
    # open a csv file 
    # Write the xml file into a csv file
    with open(PATH_CSV_TEST, 'w', encoding='utf-8') as fhandle:
        writer = csv.writer(fhandle)
        # The column names of the csv file
        writer.writerow(('Movie', 'Review_id', 'Name', 'User_id','Commentaire'))

        # Extract texts from the tags
        for comment in soup.find_all('comment'):
            writer.writerow((comment.movie.text,
                             comment.review_id.text,
                             comment.find('name').text,
                             comment.user_id.text, 
                             comment.commentaire.text))
