import re
import pickle

tfidf = pickle.load(open("model/tfidf.pkl", "rb"))


def cleanResume(txt):
    txt = re.sub("http\S+\s", " ", txt)
    txt = re.sub("RT|cc", " ", txt)
    txt = re.sub("#\S+\s", " ", txt)
    txt = re.sub("@\S+", "  ", txt)
    txt = re.sub("[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", txt)
    txt = re.sub(r"[^\x00-\x7f]", " ", txt)
    txt = re.sub("\s+", " ", txt)
    return txt
