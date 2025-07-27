import os

# مسیر فایل ورودی (فایل YAML فعلی شما)
input_file_path = './desktop_gcn_nasbench201.yaml'

# مسیر فایل خروجی (همون فایل، با کدگذاری جدید)
output_file_path = './desktop_gcn_nasbench201.yaml'

# کدگذاری فعلی که احتمالا مشکل داره (معمولا cp1252 در ویندوز)
# current_encoding = 'cp1252' # یا 'latin-1' یا 'iso-8859-1' رو امتحان کنید اگر cp1252 جواب نداد
current_encoding = 'latin-1'
# کدگذاری مورد نظر (UTF-8)
target_encoding = 'utf-8'

try:
    # فایل رو با کدگذاری فعلی بخون
    with open(input_file_path, 'r', encoding=current_encoding) as f_in:
        content = f_in.read()

    # فایل رو با کدگذاری UTF-8 بنویس
    with open(output_file_path, 'w', encoding=target_encoding) as f_out:
        f_out.write(content)

    print(f"فایل '{input_file_path}' با موفقیت به '{target_encoding}' تبدیل و ذخیره شد.")

except UnicodeDecodeError as e:
    print(f"خطا در خواندن فایل با کدگذاری '{current_encoding}': {e}")
    print("لطفاً کدگذاری 'current_encoding' در اسکریپت را تغییر دهید (مثلاً به 'latin-1' یا 'iso-8859-1').")
except FileNotFoundError:
    print(f"خطا: فایل '{input_file_path}' پیدا نشد. لطفاً مسیر را بررسی کنید.")
except Exception as e:
    print(f"خطای ناشناخته: {e}")