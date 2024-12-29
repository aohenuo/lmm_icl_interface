from .prompt_template import PromptTemplate


class LMMPromptManager:
    def __init__(
        self,
        ice_prompt_template,
        column_token_map,
        label_field,
        sep_token="",
        query_prompt_template=None,
    ):
        """
        @ice_prompt_template:
        """
        if query_prompt_template is None:
            query_prompt_template = ice_prompt_template

        if not isinstance(ice_prompt_template, str):
            ice_prompt_template = dict(ice_prompt_template)
        if not isinstance(query_prompt_template, str):
            query_prompt_template = dict(query_prompt_template)
        self.ice_pt = PromptTemplate(
            ice_prompt_template,
            column_token_map=dict(column_token_map),
        )
        self.query_pt = PromptTemplate(
            query_prompt_template,
            column_token_map=dict(column_token_map),
        )
        self.label_field = label_field
        self.sep_token = sep_token

    def gen_ice_text_with_label(self, item, label=None, add_sep_token=True):
        if label is None and isinstance(self.ice_pt.template, dict):
            label = item[self.label_field]
        prompt = self.ice_pt.generate_ice_item(item, label)
        if add_sep_token:
            prompt = prompt + self.sep_token
        return prompt

    def get_only_ice_label(self, item, label=None):
        return subtract_strings(self.gen_ice_text_with_label(item, label, add_sep_token=True), self.gen_query_text_without_label(item))
    
    def gen_query_text_with_label(self, item, label=None, add_sep_token=True):
        if label is None and isinstance(self.query_pt.template, dict):
            label = item[self.label_field]
        prompt = self.query_pt.generate_ice_item(item, label)
        if add_sep_token:
            prompt = prompt + self.sep_token
        return prompt
    
    def gen_query_text_without_label(self, item):
        return self.query_pt.generate_item(item, output_field=self.label_field)

def subtract_strings(ori_str, sub_str):
    if sub_str not in ori_str:
        return ori_str
    else:
        new_str = ori_str.replace(sub_str, "")
        new_str = new_str.strip()
        return new_str