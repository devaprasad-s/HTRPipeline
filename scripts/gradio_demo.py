import json

import cv2
import gradio as gr

from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree

with open('data/words_alpha.txt') as f:
    word_list = [w.strip().upper() for w in f.readlines()]
prefix_tree = PrefixTree(word_list)


def process_page(img, scale, margin):
    # read page
    read_lines = read_page(img,
                           detector_config=DetectorConfig(scale=scale, margin=margin),
                           line_clustering_config=LineClusteringConfig(min_words_per_line=2),
                           reader_config=ReaderConfig(decoder='word_beam_search',
                                                      prefix_tree=prefix_tree))

    # create text to show
    res = ''
    for read_line in read_lines:
        res += ' '.join(read_word.text for read_word in read_line) + '\n'

    # create visualization to show
    for i, read_line in enumerate(read_lines):
        for read_word in read_line:
            aabb = read_word.aabb
            cv2.rectangle(img,
                          (aabb.xmin, aabb.ymin),
                          (aabb.xmin + aabb.width, aabb.ymin + aabb.height),
                          (255, 0, 0),
                          2)
            cv2.putText(img,
                        read_word.text,
                        (aabb.xmin, aabb.ymin + aabb.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color=(255, 0, 0))

    return res


with open('data/config.json') as f:
    config = json.load(f)

examples = []
for k, v in config.items():
    examples.append([f'data/{k}', v['scale'], v['margin'], False, 2, v['text_scale']])

# define gradio interface
gr.Interface(fn=process_page,
             inputs=[gr.Image(label='Input image',show_label=False),
                     gr.Slider(0, 10, 1, step=0.01, label='Scale'),
                     gr.Slider(0, 25, 1, step=1, label='Margin'),],
             outputs=gr.Textbox(label='Read Text'),
             allow_flagging='never',
             title='Person Identification Using Handwriting Analysis',
             theme=gr.themes.Monochrome()).launch()
