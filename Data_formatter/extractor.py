# Import the necessary libraries
from bs4 import BeautifulSoup
import pandas as pd
import csv
from constants import PATH_CSV_TRAIN, PATH_CSV_VALID, PATH_XML_TRAIN, PATH_XML_VALID



def retrieve_data(path_xml,path_csv):

    # Open and read the xml file with beautifulSoup class: The encoding of the file  is utf-8
    with open(path_xml, 'r', encoding='utf-8') as fhandle:
        soup = BeautifulSoup(fhandle, 'xml')
    

    # open a csv file 
    # Write the xml file into a csv file
    with open(path_csv, 'w', encoding='utf-8') as fhandle:
        writer = csv.writer(fhandle)
        # The column names of the csv file
        writer.writerow(('Movie', 'Review_id', 'Name', 'User_id','Note','Commentaire'))

        # Extract texts from the tags
        for comment in soup.find_all('comment'):
            writer.writerow((comment.movie.text,
                             comment.review_id.text,
                             comment.find('name').text,
                             comment.user_id.text, 
                             comment.note.text, 
                             comment.commentaire.text))


if __name__=='__main__':

    # To convert XML file into csv file for the train data
    retrieve_data(PATH_XML_TRAIN, PATH_CSV_TRAIN)

    # To convert XML file into csv file for the validation data
    retrieve_data(PATH_XML_VALID, PATH_CSV_VALID)





        