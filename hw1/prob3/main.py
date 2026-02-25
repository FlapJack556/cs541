import submission, util
import re
from collections import defaultdict

# Read in examples
trainExamples = util.readExamples('names.train')
devExamples = util.readExamples('names.dev')


def featureExtractor(x):  # phi(x)
    # x = "took Mauritius into"
    phi = defaultdict(float)
    tokens = x.split()
    left, entity, right = tokens[0], " ".join(tokens[1:-1]), tokens[-1]
    entity_tokens = entity.split()  
    # given features
    phi['entity is ' + entity] = 1
    phi['left is ' + left] = 1
    phi['right is ' + right] = 1

    # per-word binary features
    for word in entity_tokens:
        phi['entity word: ' + word] = 1

    # prefix/suffix lengths
    for word in entity_tokens:
        w = word.lower()
        for n in [2, 3, 4]:
            if len(w) >= n:
                phi['pre{}:{}'.format(n, w[:n])] = 1
                phi['suf{}:{}'.format(n, w[-n:])] = 1  

    # capitalisation
    phi['all_caps'] = float(all(w.isupper() for w in entity_tokens))
    phi['title_case'] = float(
        all(w[0].isupper() and (len(w) < 2 or not w[1:].isupper())
            for w in entity_tokens)
    )
    phi['first_cap'] = float(entity_tokens[0][0].isupper())  
    phi['starts_lower'] = float(entity_tokens[0][0].islower())

    # initial patterns
    phi['has_initial'] = float(
        any(re.match(r'^[A-Z]\.$', w) for w in entity_tokens)
    )

    # entity length
    n_words = len(entity_tokens)
    phi['num_words'] = n_words
    phi['1_word'] = float(n_words == 1)
    phi['2_words'] = float(n_words == 2)
    phi['3_words'] = float(n_words == 3)
    phi['4plus_words'] = float(n_words >= 4)

    # non-name characters
    phi['has_digit'] = float(any(c.isdigit() for c in entity))  
    phi['has_hyphen'] = float('-' in entity)
    phi['all_alpha'] = float(all(c.isalpha() or c.isspace() for c in entity))
    phi['has_date'] = float(bool(re.search(r'\d{4}-\d{2}-\d{2}', entity)))
    phi['has_slash'] = float('/' in entity)

    # right context
    phi['right_open_paren'] = float(right == '(')
    phi['right_close_paren'] = float(right == ')')
    phi['right_is_b'] = float(right == 'b')
    phi['right_is_c'] = float(right == 'c')
    phi['right_is_date'] = float(bool(re.match(r'^\d{4}-\d{2}-\d{2}$', right)))
    phi['right_is_number'] = float(right.isdigit())

    speaker_verbs = {
        'said', 'told', 'added', 'denied', 'warned', 'noted', 'says',
        'wrote', 'argued', 'claimed', 'announced', 'confirmed', 'declared',
        'explained', 'stated', 'admitted', 'replied', 'responded',
        'suggested', 'urged', 'praised', 'lamented', 'blamed', 'criticised',
        'criticized', 'acknowledged', 'insisted', 'stressed', 'pledged',
        'promised', 'predicted', 'repeated', 'reported'
    }
    phi['right_speaker_verb'] = float(right in speaker_verbs)

    phi['right_person_verb'] = float(right in {
        'was', 'has', 'is', 'will', 'had', 'hit', 'gave', 'scored',
        'won', 'lost', 'beat', 'shot', 'fired', 'ran', 'led', 'did'
    })

    # left context
    phi['paren_wrap'] = float(left == '(' and right == ')')

    titles = {
        # Political / government
        'president', 'minister', 'prime', 'premier', 'general', 'secretary',
        'secretary-general', 'vice-president', 'chancellor', 'senator',
        'governor', 'ambassador', 'commissioner', 'prosecutor', 'judge',
        'justice', 'director', 'director-general', 'chairman',
        'speaker', 'state', 'finance', 'affairs', 'col', 'colonel', 'major',
        'sergeant', 'captain', 'mr', 'mr.', 'mrs.', 'ms.', 'sir',
        # Royalty / religious
        'king', 'queen', 'prince', 'princess', 'sheikh', 'patriarch',
        'archbishop', 'bishop', 'lord', 'rev.', 'father', 'sister', 'brother',
        # Academic / professional
        'professor', 'prof.', 'doctor', 'dr', 'dr.', 'lawyer', 'detective',
        # Sports roles / seedings
        'champion', 'seed', 'top-seeded', 'fifth-seed', 'third-seeded',
        'seeded', 'finalist', 'runner-up', 'winner', 'coach', 'manager',
        'midfielder', 'striker', 'defender', 'forward', 'winger', 'goalkeeper',
        'singles', 'doubles', 'left-back', 'centre', 'lock', 'batsman',
        # Nationality / demonym words
        'frenchman', 'spaniard', 'swede', 'austrian', 'german', 'american',
        'african', 'namibian', 'indonesian', 'canadian', 'belgian', 'moroccan',
        'briton', 'czech', 'argentine', 'australian', 'polish', 'israeli',
        'iranian', 'russian', 'chinese', 'japanese', 'korean', 'brazilian',
        # Misc roles
        'spokesman', 'spokeswoman', 'businessman', 'policeman', 'strongman',
        'actor', 'aide', 'adviser', 'official', 'chief', 'head', 'leader',
        'member', 'veteran', 'teenager', 'fellow', 'fellow-american',
    }
    phi['left_is_title'] = float(left.lower() in titles)

    phi['left_beat'] = float(left.lower() == 'beat')
    phi['left_ranked_num'] = float(bool(re.match(r'^\d+\.$', left)))
    phi['score_context'] = float(left.isdigit() and right.isdigit())
    phi['rank_before_paren'] = float(left.isdigit() and right == '(')
    phi['left_the'] = float(left.lower() == 'the')
    phi['left_is_b'] = float(left == 'b')

    # combined features
    phi['title_case_before_paren'] = float(phi['title_case'] and right == '(')
    phi['title_plus_name'] = float(phi['left_is_title'] and phi['title_case'])
    phi['caps_paren_wrap'] = float(phi['all_caps'] and phi['paren_wrap'])
    phi['initial_before_paren'] = float(phi['has_initial'] and right == '(')

    particles = {'de', 'van', 'von', 'del', 'der', 'le', 'la', 'di', 'da'}
    phi['has_particle'] = float(
        any(w.lower() in particles for w in entity_tokens)
    )

    return phi


# Learn a predictor
weights = submission.learnPredictor(trainExamples, devExamples, featureExtractor, 10, 0.1)
util.outputWeights(weights, 'weights')
util.outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')

# Test!!!
testExamples = util.readExamples('names.test')
predictor = lambda x: 1 if util.dotProduct(featureExtractor(x), weights) > 0 else -1
print('test error =', util.evaluatePredictor(testExamples, predictor))