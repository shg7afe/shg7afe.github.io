[[Back to index]](./index.html)

Before we start to examine the keywords that have been emerging in the NSF Grant data set in the last few years, we do some preprocessing on the abstract texts with the [spaCy NLP library](https://spacy.io).


```python
import re, spacy
```

Load the standard spaCy English pipeline and disable some components we do not need (e.g., named entity recognition).


```python
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
print(nlp.pipe_names)
```

    ['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer']


Add some stop words to spaCy's default list (these appear in some HTML tags in the abstracts).


```python
for sw in ['lt', 'gt', 'br']:
    nlp.Defaults.stop_words.add(sw)
    nlp.vocab[sw].is_stop = True
```

We create a preprocessing function that converts the abstracts to lowercase and replaces all non-alphanumeric characters with spaces.


```python
pattern_anum = re.compile('[^a-z0-9]+')

def pre1(text):
    return pattern_anum.sub(r' ', text.lower())
```

We will process the data in chunks in order to not run out of memory. The `create_csv` function writes a header-only CSV file to disk and then passes data in chunks from the original NSF Grant data file to the `process_text` function. The `process_text` function removes all stop words and converts all plural words to singular with the spaCy lemmatizer, and appends the processed text to the CSV file. 


```python
lemma_tags = {"NNS", "NNPS"}
batchsize = 256
outfile = './nsf-sg-spacy-cleaned.csv'

def process_text(chunk, batchsize, outfile):
    chunk_text = chunk.abstract.map(pre1)
    
    docproc = []    
    for doc in nlp.pipe(chunk_text, batch_size=batchsize):
        t = [token for token in doc if not token.is_stop and not token.is_digit]        
        t = [token.text if token.tag_ not in lemma_tags else token.lemma_ for token in t]
        docproc.append(' '.join(t))
    
    chunk['abstract'] = docproc
    chunk.to_csv(outfile, header=None, mode='a')

def create_csv(outfile):
    df = pd.DataFrame(columns=['year', 'instrument', 'directorate', 'division', 'abstract', 'funding'])
    df.to_csv(outfile, encoding='utf-8')

    for chunk in pd.read_csv('./nsf-standard-grants.csv.xz', encoding='utf-8', index_col=0, chunksize=batchsize):
        print(chunk.index[0])
        process_text(chunk, batchsize, outfile)
```

Running the conversion for the whole dataset takes a while...


```python
create_csv(outfile)
```

After the processing is complete, [we can try to identify some keywords that have been emerging in the last few years.](./topwords.html)
 
[[Back to index]](./index.html)
