import numpy as np
import pandas as pd
import os


def parse_ndk():    
    raw_ndk = ""
    with open(f'{os.path.dirname(__file__)}/raw_data/jan76_dec20.ndk') as f:
        raw_ndk = f.read()

    raw_ndk = raw_ndk.rstrip("\n").split("\n")

    assert len(raw_ndk) % 5 == 0, "The number of lines in the raw_ndk file is not divisible by 5"

    num_events = len(raw_ndk)//5

    events_raw = []
    for num_iter in range(num_events):
        # date: 1,6-15
        # location: 1,57-80
        # exponent: 4, 1-2
        # scalar moment: 5, 50-56
        # Reference: https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/allorder.ndk_explained
        # 
        # Mw = 2/3 * (lgM0  - 16.1), 
        # Reference: https://www.globalcmt.org/CMTsearch.html
        an_event = raw_ndk[num_iter*5:num_iter*5+5]
        date = an_event[0][5:16].strip()
        location = an_event[0][56:81].strip()
        exponent = float(an_event[3][0:2].strip())
        scalar = float(an_event[4][49:57].strip())
        mw = float(2/3* (np.log10(scalar) + exponent - 16.1))
        events_raw.append([date, location, scalar, exponent, mw])
    df = pd.DataFrame(events_raw, columns=["date", "location", "scalar", "exponent", "Mw"]).set_index('date')
    return df

if __name__ == "__main__":
    df = parse_ndk()
    print(df.to_string())
    #                               location  scalar  exponent        Mw
    #   date                                                            
    #   1976/01/01   KERMADEC ISLANDS REGION   9.560      26.0  7.253639
    #   1976/01/05                      PERU   3.790      24.0  5.652426
    #   1976/01/06  OFF EAST COAST OF KAMCHA   1.980      25.0  6.131110
    #   1976/01/09           VANUATU ISLANDS   3.640      25.0  6.307401
    #   1976/01/13            ICELAND REGION   3.300      25.0  6.279009
    #   1976/01/14          KERMADEC ISLANDS   6.020      27.0  7.786398
    #   1976/01/14   KERMADEC ISLANDS REGION   8.180      27.0  7.875169
    #   1976/01/14   KERMADEC ISLANDS REGION   4.540      25.0  6.371371
    #   1976/01/15   KERMADEC ISLANDS REGION   1.960      25.0  6.128171
    #   1976/01/15          KERMADEC ISLANDS   9.630      24.0  5.922418
    #   1976/01/21             KURIL ISLANDS   6.910      26.0  7.159652
    #   1976/01/23                FLORES SEA   1.580      26.0  6.732438
    #   1976/01/24   KERMADEC ISLANDS REGION   2.560      25.0  6.205493
    #   1976/01/26   KERMADEC ISLANDS REGION   1.270      24.0  5.335869
    # ...