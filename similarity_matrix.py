from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np
from itertools import combinations
from collections import OrderedDict

class SimilarityCount(MRJob):
    
    def mapper_1(self, key, line):
        user_id, item_id, rate, timestamp = line.split(':')
        yield int(user_id), (item_id, float(rate))
    
    def reducer_1(self, user_id, values):
        total = 0
        items = []
        rates = []
        for item_id, rating in values:
            total += 1
            items.append(int(item_id))
            rates.append(int(rating))
        avg_rate = float(np.sum(rates) / float(total))

        indexes = list(range(len(items)))
        for i, j in combinations(indexes, 2):
            yield (items[i], items[j]), (user_id, rates[i], rates[j], avg_rate)

    def reducer_2(self, keys, values):
        sum_above = 0
        sqr_i = 0
        sqr_j = 0
        for u, i, j, r in values:
            sum_above += (i-r)*(j-r)
            sqr_i += (i-r)**2
            sqr_j += (j-r)**2
        if sqr_i != 0 and sqr_j != 0:
            result = sum_above / (sqr_i**0.5 * sqr_j**0.5)
        else:
            result = 0
        if result < 0:
            result = 0
        yield '|'.join([str(keys[0]), str(keys[1])]), result


    def steps(self):
        return [
            MRStep(mapper=self.mapper_1,
                   reducer=self.reducer_1),
            MRStep(reducer=self.reducer_2)
            ]
if __name__ == '__main__':
    SimilarityCount.run()
