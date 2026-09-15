import torch
from transformers import AutoModelForSeq2SeqLM
from indobenchmark import IndoNLGTokenizer
import shutil, os

_original_pad = IndoNLGTokenizer.pad
def _patched_pad(self, *args, **kwargs):
    kwargs.pop("padding_side", None)
    return _original_pad(self, *args, **kwargs)
IndoNLGTokenizer.pad = _patched_pad

LANG_ID = 40002  # [indonesian]

_original_decode = IndoNLGTokenizer.decode
def _patched_decode(self, *args, **kwargs):
    kwargs.pop("clean_up_tokenization_spaces", None)
    return _original_decode(self, *args, **kwargs)
IndoNLGTokenizer.decode = _patched_decode

mname = "/content/drive/MyDrive/indobart-liputan6-finetuned"  
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = IndoNLGTokenizer.from_pretrained(mname)
model = AutoModelForSeq2SeqLM.from_pretrained(mname).to(device)
model.config.decoder_start_token_id = LANG_ID
model.eval()

article = "Liputan6 . com , Jakarta : Pemerintah masih memberikan waktu dua minggu lagi kepada seluruh konglomerat yang telah menandatangani perjanjian pengembalian bantuan likuiditas Bank Indonesia dengan jaminan aset ( MSAA ) , untuk secepatnya menyerahkan jaminan pribadi serta aset . Jika lewat dari tenggat tersebut , pemerintah akan menerapkan tindakan hukum . Hal tersebut dikemukakan Menteri Koordinator Bidang Perekonomian Rizal Ramli di Jakarta , baru-baru ini . Rizal mengakui bahwa permintaan untuk meminta jaminan pribadi atau personal guarantee pada awalnya ditentang sejumlah konglomerat . Sebab para debitor menganggap tindakan tersebut memungkinkan pemerintah untuk menyita seluruh aset mereka baik yang berada di dalam maupun luar negeri . Sejauh ini , penilaian jaminan MSAA baru dilakukan atas aset milik Grup Salim . Tetapi , nilai aset yang dijaminkan Kelompok Salim atas utang BLBI Bank Central Asia diperkirakan tak lebih dari Rp 20 triliun . Padahal , kewajiban mereka mencapai Rp 52 triliun . Sementara itu , pemerintah dengan DPR sepakat , hingga akhir Oktober mendatang , para konglomerat penandatangan MSAA harus sudah menutupi kekurangan mereka dengan menyerahkan aset baru . Selain itu , para pengutang tersebut diwajibkan memberikan jaminan pribadinya . ( TNA/Merdi Sofansyah dan Anto Susanto ) ."

max_src_len = 1024
ids = tokenizer.encode(article.lower(), max_length=max_src_len - 3, truncation=True, add_special_tokens=False)
ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id, LANG_ID]

input_ids = torch.tensor([ids]).to(device)
attention_mask = torch.tensor([[1] * len(ids)]).to(device)

# generate summary for sanity check
with torch.no_grad():
    summary_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=102,
        min_length=21,
        num_beams=4,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        early_stopping=True,
    )

summary = tokenizer.decode(summary_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("SUMMARY:", summary)