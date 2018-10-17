import json
from classification_process import Document
from text_parser import Word_Counter

tokenizer = Word_Counter()

class Movie(Document):
    def __init__(self, movie_json, tokenizer = tokenizer):
        title = movie_json['Title']
        categories = [x.strip() for x in movie_json['Genre'].split(',')]
        plot = movie_json['Plot']

        term_freq = tokenizer.parse(plot)
        super(Movie, self).__init__(title=title, categories=categories, term_freq=term_freq)

    def _get_genres(self):
        return self.categories
    genres = property(_get_genres)

    def _get_plot(self):
        return self.text

    def is_in_genre(self, genre):
        for g in self.genres:
            if g == genre:
                return True
        return False

    @staticmethod
    def movies_in_file(f):
        for line in f:
            movie = Movie(json.loads(line))
            yield movie

    @staticmethod
    def all_movies_in_file(infile):
        with open(infile, 'r') as f:
            return list(Movie.movies_in_file(f))