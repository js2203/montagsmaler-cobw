import os,logging
from gensim.models import word2vec
from flask import Flask
from flask import request

application = Flask(__name__)

model = None

class Sentences(object):
    def __init__(self, dirnameP):
        self.dirnameP = dirnameP

    def __iter__(self):
        for line in open(self.dirnameP, encoding="utf8"):
            linelist = line.split()
            if len(linelist) > 3 and linelist[0][0] != "<":
                yield [w.lower().strip(",."" \" () :; ! ?") for w in
                       linelist]

try:
    print('start loading model')
    model = word2vec.Word2VecKeyedVectors.load_word2vec_format(
        os.path.join(os.path.dirname(__file__),'model.bin.gz'),
        binary=True)
    model.init_sims(replace=True)
    print("Already existing model is loaded")
except Exception as e:
    print("Error: {}".format(e))


@application.route('/', methods=['GET'])
def health_check():
    if not model:
        return 'FAILURE'
    else:
        return 'SUCCESS'


@application.route('/word_similarity', methods=['GET'])
def word_similarity():
    try:
        similarity = model.similarity(request.args.get('base'), 
                                      request.args.get('compare'))
        return "{}".format(similarity)
    except:
        return 'error at calculating similarities'

if __name__ == "__main__":
    application.run(use_reloader=False)