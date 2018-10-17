from pickle import PickleCache
from text_parser import Word_Counter
from classification_process import MultiCategoryCorpus, NaiveBayesClassifier
from movie_class import Movie


tokenizer = Word_Counter()
classifier = NaiveBayesClassifier

training_dataset = 'train.json'
test_dataset = 'test.json'


class MovieCache(PickleCache):
    def __init__(self, pickle_suffix='.pickle'):
        super(MovieCache, self).__init__(pickle_suffix=pickle_suffix)

    def load_object(self, fpath):
        return Movie.all_movies_in_file(fpath)


def load_all_movies(movie_files=[training_dataset, test_dataset], clear_cache=False):
    cache = MovieCache()
    return [cache.read(mf, clear_cache) for mf in movie_files]


def test(clear_cache=False, classifier_class=classifier, verbose=True):
    """Loads the training and test datasets, then trains and tests it."""

    print('loading')
    M = load_all_movies(movie_files=[training_dataset, test_dataset], clear_cache=clear_cache)
    (mtrain, mtest) = (M[0], M[1])
    print('building corpus')
    corpus = MultiCategoryCorpus.build(mtrain)
    print('training')
    classifier = classifier_class(corpus)
    train_stats = classifier.train(mtrain)
    print('testing')
    test_stats = classifier.test(mtest)
    results = {'mtrain': mtrain,
               'mtest': mtest,
               'corpus': corpus,
               'classifier': classifier,
               'strain': train_stats,
               'stest': test_stats,
               }

    if verbose: test_stats.print_stats()
    return results
if __name__ == '__main__':
    test()

