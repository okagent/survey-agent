import fitz
import re
import numpy as np

class Paper:
    def __init__(self, path):
        """
        初始化函数，根据pdf路径初始化Paper对象 
        """
                       
        self.path = path

        self.abs = abs
        self.title_page = 0
        
        self.roman_num = ["I", "II", 'III', "IV", "V", "VI", "VII", "VIII", "IIX", "IX", "X"]
        self.digit_num = [str(d+1) for d in range(10)]
        self.candidate_refer_section = set([
            "introduction", "conclusion", "conclusions", "conclusion and future work",
            "reference", "references", "bibliography", "acknowledgments", "acknowledgements", "appendix"
        ]) # 可能前面不会有序号的section标题
        self.pdf = fitz.open(self.path)
        self.block_list = [[block for block in page.get_text("dict")["blocks"] if block["type"] == 0] for page in self.pdf]
        self.avg_size = self.get_average_line_size()
        self.section_names = self.get_chapter_names()   # 段落标题（得到section标题）
        self.parse_pdf()  # 得到每个section对应的内容
      
    def parse_pdf(self):
        """
        完整处理流程 
        """
        self.text_list = [page.get_text() for page in self.pdf]
        self.section_page_dict = self._get_all_page_index() # 段落与页码的对应字典
        self.section_text_dict = self._get_all_page() # 段落与内容的对应字典
        self.pdf.close()       
        self.get_chapter_needed()   
    
    def get_average_line_size(self):
        """
        计算全篇平均span size
        """
        total_size_sum = 0
        total_span_count = 0

        for page in self.pdf:
            blocks = [
                block for block in page.get_text("dict")["blocks"]
                if block['type'] == 0
            ]

            for block in blocks:
                for line in block['lines']:
                    line_size_sum = sum(span['size'] for span in line['spans'])
                    line_size_count = len(line['spans'])
                    if line_size_count > 0:
                        total_size_sum += line_size_sum
                        total_span_count += line_size_count

        if total_span_count == 0:
            return 0  # 避免除以零的错误
        return total_size_sum / total_span_count
    
    def find_sections_in_text(self, text, section_name):
        """
        定义一个函数 在文中找到匹配的标题位置
        """
        roman_numeral = r'(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))'
        pattern = r'^(?:' + roman_numeral + r'\.? *|\d+\.? *|\w\.? *| *|)' + re.escape(section_name) + r'\s*$' 
        section_name_upper = section_name.upper()
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        index = -1
        if matches:
            # 取得最后一个匹配项的索引
            last_match = matches[-1]
            inpage_index = last_match.start()
            index = inpage_index
        else:
            pattern = r'^(?:' + roman_numeral + r'\.? *|\d+\.? *|\w\.? *| *|)' + re.escape(section_name_upper) + r'\s*$' 
            # 使用正则表达式进行匹配
            matches = list(re.finditer(pattern, text))
            if matches:
                last_match = matches[-1]
                inpage_index = last_match.start()
                index = inpage_index
        if index == -1:
            pattern = re.escape(section_name) + r'[\s\n]+'
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            if matches:
                last_match = matches[-1]
                inpage_index = last_match.start()
                index = inpage_index
            else:
                pattern = re.escape(section_name_upper) + r'[\s\n]+'
                matches = list(re.finditer(pattern, text, re.MULTILINE))
                if matches:
                    last_match = matches[-1]
                    inpage_index = last_match.start()
                    index = inpage_index

        return index
          
    def get_chapter_names(self):
        """
        定义一个函数，识别每个章节名称，并返回一个列表
        
        Returns:
            chapter_names 
        """
        chapter_names = []
        conclude_flag = False
        
        for page in self.block_list:
            for block in page:
                
                # 计算这块block的平均span size
                total_size_sum = 0
                total_span_count = 0
                for line in block['lines']:
                        line_size_sum = sum(span['size'] for span in line['spans'])
                        line_size_count = len(line['spans'])
                        if line_size_count > 0:
                            total_size_sum += line_size_sum
                            total_span_count += line_size_count
                        
                if total_span_count == 0:
                    mean_size = 0  # 避免除以零的错误
                else:
                    mean_size = total_size_sum / total_span_count
                
                cur_text = self.split_block(block).replace('\u2003', ' ')
                cur_text_list = cur_text.split('\n')
                for i,line in enumerate(cur_text_list):
                    line = line.strip()
                    
                    point_split_list = line.split('.')
                    space_split_list = line.split(' ')
                    if len(space_split_list) < 10 and len(line) > 1 and 'arXiv' not in line and 'http' not in line:
                        if len(point_split_list) < 5 and ("Introduction" in line or "Conclusion" in line or "INTRODUCTION" in line or "CONCLUSION" in line):
                            if line in ''.join(chapter_names):
                                continue
                            else:
                                chapter_names.append(line)
                            if "conclusion" in line.lower():
                                conclude_flag = True
                    
                        elif len(point_split_list) < 5 and any(char.isalpha() for char in line) and (point_split_list[0] in self.roman_num or point_split_list[0] in self.digit_num or \
                            space_split_list[0] in self.roman_num or space_split_list[0] in self.digit_num):
                            chapter_names.append(line)
                            continue
                        
                        if line.lower() in self.candidate_refer_section:
                            chapter_names.append(line)
                            continue
                        
                        # 1.2是拍的倍数 用来判断比全篇平均span size都大1.2倍的行 一般正文的section都可以考前面的逻辑得到
                        if any(char.isalpha() for char in line) and mean_size > 1.2 * self.avg_size:

                            if conclude_flag:
                                chapter_names.append(line)
                                continue

        self.section_list = list(set(chapter_names)) # 去重
        
        return chapter_names

    def _get_all_page_index(self):
        """
        定义需要寻找的章节名称列表
        初始化一个字典来存储找到的章节和它们在文档中出现的页码
        
        Returns:
            section_page_dict 每个section及其对应出现的名字
        """
        section_page_dict = {}
        section_encounter = {}
        # 遍历每一页文档
        for page_index, page in enumerate(self.block_list):
            # 获取当前页面的文本内容
            cur_text = '\n'.join([self.split_block(block) for block in page]).replace('\u2003', ' ')
            
            # 遍历需要寻找的章节名称列表
            for section_name in self.section_list:
                # conclusion及之后的模块都找至多第二个匹配的
                if "introduction" in section_name.lower() and section_page_dict.get(section_name) is not None:
                    continue
                elif section_encounter.get(section_name) is not None:
                    continue

                # 将章节名称转换成大写形式
                # 如果当前页面包含"Abstract"这个关键词
                
                if "Abstract" == section_name and section_name in cur_text:
                    # 将"Abstract"和它所在的页码加入字典中
                    inpage_index = cur_text.find(section_name)
                    section_page_dict[section_name] = (page_index, inpage_index)
                
                # 如果当前页面包含章节名称，则将章节名称和它所在的页码、在这页的index加入字典中
                else:
                    inpage_index = self.find_sections_in_text(cur_text, section_name)
                    if inpage_index != -1:
                        if section_page_dict.get(section_name) is not None:
                            section_encounter[section_name] = 1
                            section_page_dict[section_name] = (page_index, inpage_index)
                        else:
                            section_page_dict[section_name] = (page_index, inpage_index)
        # 返回所有找到的章节名称及它们在文档中出现的页码
        return section_page_dict

    def split_block(self, block):
        """
        根据每一行的长度判断block中每一行是否需要换行
        
        Returns:
            block_text 理好的该block文字 
        """
        text_list = block['lines']
        block_text = ''
        
        for i, line in enumerate(text_list):
            text = ''.join(span['text'] for span in line['spans'])
            if text.strip().lower() in self.candidate_refer_section:
                block_text += text + '\n'
                continue
            if "introduction" in text.lower() and i < len(text_list)-1: # 防止introduction底下有一个首字母很大的情况
                net_text = ''.join(span['text'] for span in text_list[i+1]['spans'])
                if len(net_text.strip())==1:
                    block_text += text + '\n'
                    continue
            x2_condition = line['bbox'][2] < block['bbox'][0] + 0.9*(block['bbox'][2]-block['bbox'][0])
            
            if i == len(text_list)-1:
                y1_condition = True
            else:
                y1_condition = line['bbox'][3] < text_list[i+1]['bbox'][1]
            if x2_condition and y1_condition:
                # 不需要结尾是句号等（防止标题被误伤）
                block_text += text + '\n'
            elif text.endswith('-'):
                block_text += text[:-1]
            else:
                block_text += text + ' '
        return block_text.strip()
    
    def find_reference_or_acknowledgement_section(self, section_page_dict_list):
        """
        在section列表中找到最后一个包含"References"或者"Reference"或者"Bibliography的section，
        如果没有找到，则找包含"Acknowledgements"的section。
        还没找到再找"Appendix"的section。将找到的section及其后面的部分删去。
        
        param section_page_dict_list: 一个包含section名称的列表
        Returns: 找到的section名称，或者None
        """
        
        ref_section = None
        ack_section = None
        app_section = None

        for section in section_page_dict_list:
            section_name = section.lower()  # 假设section的名称是每个元组的第一个元素
            if "reference" in section_name or 'bibliography' in section_name:
                ref_section = section  # 更新为最后一个包含“References”的section
            elif ("acknowledgements" in section_name or "acknowledgments" in section_name) and ref_section is None:
                ack_section = section # 只有在还没有找到含有“References”的section时才记录
            elif "appendix" in section_name and ref_section is None and ack_section is None:
                app_section = section  # 只有在还没有找到含有“References”或“Acknowledgements”的section时才记录

        return ref_section if ref_section is not None else ack_section if ack_section is not None else app_section
    
    def _get_all_page(self):
        """
        获取PDF文件中每个页面的文本信息，并将文本信息按照章节组织成字典返回。

        Returns:
            section_dict (dict): 每个章节的文本信息字典，key为章节名，value为章节文本。
        """
        text_list = []
        section_dict = {}
        
        # 根据页码和页内index排序
        section_page_dict_list = [key for key, value in sorted(self.section_page_dict.items(), key=lambda x: (x[1][0], x[1][1]))]
        self.section_page_dict_list = section_page_dict_list

        # 识别reference的section
        ref_section = self.find_reference_or_acknowledgement_section(section_page_dict_list)
        self.ref_section = ref_section
        if ref_section:
            ref_page = self.section_page_dict[ref_section][0]
        else:
            ref_page = None

        # 去除reference之后的文字
        text_list = []
        for page_num, page in enumerate(self.block_list):
            if ref_page and page_num> ref_page:
                break
            page_text = '\n'.join([self.split_block(block) for block in page]).replace('\u2003', ' ')
            if ref_page and page_num == ref_page:
                start_i = self.section_page_dict[ref_section][1]
                page_text = page_text[:start_i].strip()
                if page_text:
                    page_text = re.sub(r'\d+$', '', page_text)
            text_list.append(page_text)
        self.full_text = '\n'.join(text_list)

        for sec_index, sec_name in enumerate(section_page_dict_list):
            if sec_index <= 0 and self.abs:
                continue
            else:
                # 直接考虑abstract后面 ref_section前面的内容：
                if sec_name == ref_section:
                    break
                start_page = self.section_page_dict[sec_name][0]
                if sec_index < len(section_page_dict_list)-1:
                    end_page = self.section_page_dict[section_page_dict_list[sec_index+1]][0]
                else:
                    end_page = len(text_list)

                cur_sec_text = ''
                if end_page - start_page == 0:
                    if sec_index < len(section_page_dict_list)-1:
                        next_sec = section_page_dict_list[sec_index+1]
                        start_i = self.section_page_dict[sec_name][1]
                        end_i = self.section_page_dict[next_sec][1]
                        cur_sec_text += text_list[start_page][start_i:end_i]
                else:
                    for page_i in range(start_page, end_page+1):                    
                        if page_i == start_page:
                            start_i = self.section_page_dict[sec_name][1]
                            cur_sec_text += text_list[page_i][start_i:]
                        elif page_i < end_page:
                            cur_sec_text += text_list[page_i]
                        elif page_i == end_page:
                            if sec_index < len(section_page_dict_list)-1:
                                next_sec = section_page_dict_list[sec_index+1]
                                end_i = self.section_page_dict[next_sec][1]
                                cur_sec_text += text_list[page_i][:end_i]
                section_dict[sec_name] = cur_sec_text
        return section_dict
    
    def get_chapter_needed(self):
        """
        找到Introduction和Conclusion的section 
        """
        introduction = ''
        introduction_section = ''
        for section in self.section_text_dict.keys():
            if 'introduction' in section.lower():
                introduction = self.section_text_dict[section].strip()
                introduction = re.sub(r'\d+$', '', introduction)
                introduction_section = section
                if len(introduction) <= len(section):
                    continue
                else:
                    break
        
        conclusion = ''
        for section in self.section_text_dict.keys():
            if 'conclusion' in section.lower():
                conclusion = self.section_text_dict[section].strip()
                conclusion = re.sub(r'\d+$', '', conclusion)
                if len(conclusion) == 0:
                    continue
                else:
                    break
        self.conclusion = conclusion

        # 如果introduction这个section底下有subsection 进行整合作为整体introduction
        if len(introduction_section) == len(introduction.strip()) and len(introduction_section) > 0:
            if introduction_section[0].isdigit():
                intro_sec_num = int(introduction_section[0])
                for sec in self.section_page_dict_list:
                    if 'reference' in sec.lower() or 'conclusion' in sec.lower() or 'ackowledgment' in sec.lower() or 'appendix' in sec.lower():
                        break
                    if sec.startswith(f'{intro_sec_num}.') and len(sec.split('.'))>=2:
                        if sec.split(' ')[0].split('.')[1].isdigit():
                            introduction += self.section_text_dict[sec]
        self.introduction = introduction


def validateTitle(title):
        # 将论文的乱七八糟的路径格式修正
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        new_title = re.sub(rstr, "_", title)  # 替换为下划线
        return new_title
    
    
import json
from tqdm import tqdm
def process_paper(title):
    """
    处理单篇论文
    """
    try:
        valid_title = validateTitle(title)
        paper = Paper(path=f"./files/{valid_title}.pdf")
        introduction = paper.introduction
        
        conclusion = paper.conclusion
        
        if not paper.ref_section:
            with open(f"pdf_error.log", "a+") as f:
                f.write(f"No references found in {title}.pdf")
        
        return {
            "title": title,
            "introduction": introduction,
            "conclusion": conclusion,
            "full_text": paper.full_text
        }
    except Exception as e:
        with open(f"pdf_error.log", "a+") as f:
            f.write(f"Error processing {title}: {e}\n")
        return None
    

def process_batch(batch, batch_index):
    for item in tqdm(batch, desc=f'Processing Batch {batch_index}'):
        if item.get("title"):
            result = process_paper(item.get("title"))
            if result:
                    # 更新 batch
                item.update(result)
            if "introduction" not in item:
                item["introduction"] = ""
            if "conclusion" not in item:
                item["conclusion"] = ""
            if "full_text" not in item:
                item["full_text"] = ""
    with open(f'./extract_24_1_6/processed_data_batch_{batch_index}.json', 'w', encoding='utf-8') as f:
        json.dump(batch, f, ensure_ascii=False, indent=4)

def process_papers_in_batches():
    for i in range(0, 171):
        with open(f'./unprocessed_data/unprocessed_batch_{500*i}.json', 'r', encoding='utf-8') as f:
            batch = json.load(f)
        process_batch(batch, i+1)

if __name__ == "__main__":
    process_papers_in_batches()