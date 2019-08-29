def get_model():
    from simplex_sdk import SimplexClient
    from pyltp import Segmentor
    from pyltp import Postagger
    from pyltp import NamedEntityRecognizer
    from pyltp import Parser
    from pyltp import SementicRoleLabeller
    LTP_MODEL_PATH = '/data/share/zhanghaipeng/data/ltp_model/'
    segmentor = Segmentor()
    postagger = Postagger()
    netagger = NamedEntityRecognizer()
    parser = Parser()
    labeller = SementicRoleLabeller()
    segmentor.load(LTP_MODEL_PATH+'cws.model')
    postagger.load(LTP_MODEL_PATH+'pos.model')
    netagger.load(LTP_MODEL_PATH+'ner.model')
    parser.load(LTP_MODEL_PATH+'parser.model')
    labeller.load(LTP_MODEL_PATH+'pisrl.model')
    return segmentor, postagger, netagger, parser, labeller

def test(segmentor, postagger, netagger, parser, labeller,sent):
    words = segmentor.segment(sent)
    print('分词结果:\n')
    print('\t'.join(words))

    poses = postagger.postag(words)
    print('词性标注:\n')
    for word, tag in zip(words, poses):
        print(word+'/'+tag)

    nes = netagger.recognize(words, poses)
    print('实体识别:\n')
    for word, tag in zip(words, nes):
        print(word+'/'+tag)

    arcs = parser.parse(words, poses)
    print('依存句法:\n')
    print('\t'.join('%d:%s' % (arc.head, arc.relation) for arc in arcs))

    roles = labeller.label(words, poses, arcs)
    print('语义角色:\n')
    for role in roles:
        print(str(role.index), ''.join(['%s:(%d,%d)' % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))

    segmentor.release()
    postagger.release()
    netagger.release()
    parser.release()
    labeller.release()

if __name__ == '__main__':
    segmentor, postagger, netagger, parser, labeller = get_model()
    sent = '我爱写代码。'
    sent1 = '国务院总理李克强调研上海外高桥时，提出了支持上海积极探索新机制。'
    test(segmentor, postagger, netagger, parser, labeller, sent1)
