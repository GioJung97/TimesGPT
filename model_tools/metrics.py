from cider.pyciderevalcap.ciderD.ciderD import CiderD
from cider.pyciderevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider as CiderNew
from pycocoevalcap.spice.spice import Spice
from cider.pyciderevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from cider.PyDataFormat.loadData import LoadData
from collections import defaultdict
import random, re
import string
from nltk.tokenize import word_tokenize
from lexicalrichness import LexicalRichness
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions

# Using english stopwords
stopWord = stopwords.words('english')

# Lemmatization
lemma = WordNetLemmatizer()

lexical_metrics = ['words', 'terms', 'ttr', 'rttr', 'cttr', 'msttr', 'mattr', 'mtld', 'hdd', 'vocd', 'herdan',
           'summer', 'dugast', 'maas', 'yulek', 'yulei', 'herdanvm', 'simpsond']

def generate_caption_dicts(ground_truth_captions, predicted_captions):
    gts_dict = defaultdict(list)
    predictions_dict = []
    seen_captions = {}
    for idx, caps in enumerate(ground_truth_captions):
        if isinstance(caps, list):
            print(f"caps: {caps}")
            for caption in caps:

                gts_dict[str(idx)].append({'caption': caption}) 
                # # Check if the caption has already been seen
                # if caption in seen_captions:
                #     # If seen, retrieve the original index and append to that list
                #     original_idx = seen_captions[caption]
                #     gts_dict[original_idx].append({'caption': caption})
                # else:
                #     # If not seen, add it to the current index
                #     gts_dict[str(idx)].append({'caption': caption})
                #     seen_captions[caption] = str(idx)  # Mark this caption as seen at the current index
        else:
            # Single caption case
            if caps in seen_captions:
                original_idx = seen_captions[caps]
                gts_dict[original_idx].append({'caption': caps})
            else:
                gts_dict[str(idx)].append({'caption': caps})
                seen_captions[caps] = str(idx)

    for i, caption in enumerate(predicted_captions):
        if isinstance(caption, list):
            # If caption is a list of multiple captions, create a dict for each
            for single_caption in caption:
                datapoint = {"image_id": str(i), "caption": single_caption}
                predictions_dict.append(datapoint)
        else:
            # If caption is a single string
            datapoint = {"image_id": str(i), "caption": caption}
            predictions_dict.append(datapoint)
    return gts_dict, predictions_dict

def reformat_data(old_res):
    reformatted_res = {}
    
    for datapoint in old_res:
        image_id = datapoint['image_id']
        caption = datapoint['caption']
        reformatted_res[image_id] = caption
    
    return reformatted_res

def calculate_scores(predicted_captions, ground_truth_captions):
    """
    Calculate the CIDEr score for a set of predicted captions against ground truth captions.

    :param predicted_captions: A list of predicted captions.
    :param ground_truth_captions: A list of ground truth captions.
    :return: The computed CIDEr score.
    """
    parsed_ground_truths, parsed_predictions = generate_caption_dicts(ground_truth_captions, predicted_captions)
    # print(f"parsed_ground_truths: {parsed_ground_truths}\nparsed_predictions: {parsed_predictions}")
    gts_tokenizer = PTBTokenizer('gts')
    res_tokenizer = PTBTokenizer('res')
    
    gts = gts_tokenizer.tokenize(parsed_ground_truths)
    res = res_tokenizer.tokenize(parsed_predictions)

    cider_scorer = Cider(df='coco-val')
    ciderD_scorer = CiderD(df='coco-val')

    cider_score, cider_scores = cider_scorer.compute_score(gts, res)
    ciderD_score, ciderD_scores = ciderD_scorer.compute_score(gts, res)

    # reformatting res because it needs different format for scorers below
    res = reformat_data(res)

    bleu_score, bleu_scores = Bleu(4).compute_score(gts, res)
    meteor_score, meteor_scores = Meteor().compute_score(gts, res)
    rouge_score, rouge_scores = Rouge().compute_score(gts, res)
    spice_score, spice_scores = Spice().compute_score(gts, res)
    # spice_score = None
    # spice_scores = None
    return cider_score, cider_scores, ciderD_score, ciderD_scores, bleu_score, bleu_scores, meteor_score, meteor_scores, rouge_score, rouge_scores, spice_score, spice_scores

def get_lexical_metric(text, metric='mtld'):
    lex = LexicalRichness(text)
    word_count = lex.words
    term_count = lex.terms

    if metric == 'words':
        return lex.words
    
    elif metric == 'terms':
        return lex.terms
    
    elif metric == 'ttr':
        return lex.ttr
    
    elif metric == 'rttr':
        return lex.rttr
    
    elif metric == 'cttr':
        return lex.cttr
    
    elif metric == 'msttr':
        segment_window=25   # size of each segment
        if word_count == 1: segment_window = None
        elif word_count <= 25: segment_window = word_count//2
        return lex.msttr(segment_window=segment_window)
    
    elif metric == 'mattr':
        window_size=25  # Size of each sliding window
        if word_count == 1: window_size = 1
        elif word_count <= 25: window_size = word_count//2
        return lex.mattr(window_size=window_size)
    
    elif metric == 'mtld':
        return lex.mtld(threshold=0.72)
    
    elif metric == 'hdd':
        draws=42
        if word_count == 1: draws = 1
        elif word_count <= 42: draws = word_count//2
        return lex.hdd(draws=draws)
    
    elif metric == 'vocd':
        return lex.vocd(ntokens=50, # Maximum number for the token/word size in the random samplings
                        within_sample=100, # Number of samples
                        iterations=3) # Number of times to repeat steps 1 to 3 before averaging
    
    elif metric == 'herdan':
        if word_count == 1: return None
        return lex.Herdan
    
    elif metric == 'summer':
        if word_count == 1: return None
        return lex.Summer
    
    elif metric == 'dugast':
        if term_count == word_count: return None
        return lex.Dugast
    
    elif metric == 'maas':
        if word_count == 1: return None
        return lex.Maas
    
    elif metric == 'yulek':
        return lex.yulek
    
    elif metric == 'yulei':
        return lex.yulei
    
    elif metric == 'herdanvm':
        return lex.herdanvm
    
    elif metric == 'simpsond':
        return lex.simpsond

def nlp_pipeline(text):
    
    # Lowering Text
    text = text.lower()

    # Handle Contractions
    text = contractions.fix(text)

    # Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove Non-Alphanumeric Characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Lemmatization
    text = word_tokenize(text)
    text = [lemma.lemmatize(word) for word in text]

    # Remove Stopwords                                                                                                                    
    text = " ".join([word for word in text if not word in stopWord])
    # print(text)

    return text

def computer_lexical_metrics(predicted_captions, n):
    # pre-process predicted captions
    nlp_parsed_predicted_captions = []
    for predicted_caption in predicted_captions:
        nlp_parsed_predicted_captions.append(nlp_pipeline(predicted_caption))

    # build metric reults
    metric_results = {}
    for metric in lexical_metrics:
        for _, predicted_caption in enumerate(nlp_parsed_predicted_captions):
            try:
                computed_metric = get_lexical_metric(predicted_caption, metric)
            except Exception as e:
                # error_file.write(f"Metric: {metric}\n")
                # error_file.write(f"Error occurred at index: {index}\n")
                # error_file.write(f"Text: {predicted_caption}\n")
                # error_file.write(f"Error: {e}\n\n")
                computed_metric = e
            # SHOULD I AGGREGATE SCORES PER PREDICTED CAPTION???
            if metric_results[metric] == None:
                metric_results[metric] = computed_metric
            else:
                metric_results[metric] += computed_metric

        return metric_results