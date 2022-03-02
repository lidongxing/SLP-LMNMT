import re
import os

def calc_bleu(ref, translation):
    '''Calculates bleu score and appends the report to translation
    ref: reference file path
    translation: model output file path

    Returns
    translation that the bleu score is appended to'''
    get_bleu_score = "perl multi-bleu.perl {} < {} > {}".format(ref, translation, "temp")
    os.system(get_bleu_score)
    bleu_score_report = open("temp", "r").read()
    with open(translation, "a") as fout:
        fout.write("\n{}".format(bleu_score_report))
    try:
        score = re.findall("BLEU = ([^,]+)", bleu_score_report)[0]
        new_translation = translation + "B{}".format(score)
        os.system("mv {} {}".format(translation, new_translation))
        os.remove(translation)

    except: pass
    os.remove("temp")
    return bleu_score_report

def debpe(transfile,debpefile):
    '''
    :param trans:  list
    :return:
    '''
    # sed - r
    # 's/(@@ )|(@@ ?$)//g'
    debpe_sh = "cat {} | sed -E 's/(@@ )|(@@ ?$)//g' > {}".format(transfile,debpefile)
    os.system(debpe_sh)

# debpe('devs/eval_trans_bpe.en','devs/eval_trans_debpe.en')
calc_bleu('test2010.src.en.lowcased.tc','test2010.en.trans1')
