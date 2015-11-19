class Passenger(object):

    p_id = None
    survival = None
    p_class = None
    name = None
    sex = None
    age = None
    n_sib_sp = None
    n_pa_ch = None
    ticket = None
    fare = None
    cabin = None
    embarked = None

    def __init__(self, csv_row):
        p_id = csv_row[0]
        survival = csv_row[1]
        p_class = csv_row[2]
        name = csv_row[3]
        sex = csv_row[4]
        age = csv_row[5]
        n_sib_sp = csv_row[6]
        n_pa_ch = csv_row[7]
        ticket = csv_row[8]
        fare = csv_row[9]
        cabin = csv_row[10]
        embarked = csv_row[11]
