import unittest
import numpy as np
from sentiment_model import ROOT, load_bundle, predict_sentiment, encode_review
from train import fit_tokenizer, prepare_data

class SentimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=load_bundle()

    def test_repeated_predictions_use_same_model(self):
        a=predict_sentiment('This movie was fantastic. I loved it.',self.bundle)
        b=predict_sentiment('This movie was fantastic. I loved it.',self.bundle)
        self.assertEqual(a['sentiment'],b['sentiment'])
        self.assertAlmostEqual(sum(a['scores'].values()),1.,places=5)
        self.assertEqual(set(a['scores']),{'positive','negative','neutral'})
        self.assertTrue(all(0<=score<=1 for score in a['scores'].values()))

    def test_empty_unknown_and_oversized_input(self):
        for review in ['', '   ', '!!!', 'zxqvblpptxx', 'a'*20001]:
            with self.assertRaises(ValueError):
                encode_review(review,self.bundle)

    def test_tokenizer_does_not_learn_validation_vocabulary(self):
        tokenizer=fit_tokenizer(['wonderful film','terrible acting'])
        self.assertNotIn('unseenvalidationtoken',tokenizer.word_index)
        self.assertEqual(tokenizer.texts_to_sequences(['unseenvalidationtoken']),[[tokenizer.word_index['<OOV>']]])

    def test_lfs_pointer_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError,'Git LFS pointer'):
            prepare_data(ROOT/'IMDB_reviews_dataset.csv')

    def test_streamlit_click_and_blank_validation(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        self.assertFalse(app.exception)
        app.text_area[0].input('   ')
        app.button[0].click().run()
        self.assertTrue(app.error)
        app.text_area[0].input('This movie was fantastic. I loved it.')
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(app.subheader)

if __name__=='__main__':
    unittest.main()
