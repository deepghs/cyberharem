from hcpdiff.tools.create_embedding import PTCreator

_DEFAULT_EMBEDDING_DIR = 'embs'
_DEFAULT_TRAIN_MODEL = 'deepghs/animefull-latest'


def create_embedding(
        name: str, n_words: int = 4, init_text: str = '*0.017', replace: bool = False,
        pretrained_model: str = _DEFAULT_TRAIN_MODEL, embs_dir: str = _DEFAULT_TRAIN_MODEL
):
    pt_creator = PTCreator(pretrained_model, embs_dir)
    pt_creator.creat_word_pt(name, n_words, init_text, replace)
