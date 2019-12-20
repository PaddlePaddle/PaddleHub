from paddlehub.serving.bert_serving import bert_service


class BSClient(object):
    def __init__(self,
                 module_name,
                 server,
                 max_seq_len=20,
                 show_ids=False,
                 do_lower_case=True,
                 retry=3):
        self.bs = bert_service.BertService(
            model_name=module_name,
            max_seq_len=max_seq_len,
            show_ids=show_ids,
            do_lower_case=do_lower_case,
            retry=retry)
        self.bs.add_server(server=server)

    def get_result(self, input_text):
        return self.bs.encode(input_text)
