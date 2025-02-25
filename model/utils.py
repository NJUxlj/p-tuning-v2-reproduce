from enum import Enum

from model.token_classification import (
    BertPrefixForTokenClassification,
    RobertaPrefixForTokenClassification,
    DebertaPrefixForTokenClassification,
    DebertaV2PrefixForTokenClassification
)

from model.sequence_classification import (
    BertPrefixForSequenceClassification,
    BertPromptForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    RobertaPromptForSequenceClassification,
    DebertaPrefixForSequenceClassification
)

from model.question_answering import (
    BertPrefixForQuestionAnswering,
    RobertaPrefixModelForQuestionAnswering,
    DebertaPrefixModelForQuestionAnswering
)

from model.multiple_choice import (
    BertPrefixForMultipleChoice,
    RobertaPrefixForMultipleChoice,
    DebertaPrefixForMultipleChoice,
    BertPromptForMultipleChoice,
    RobertaPromptForMultipleChoice
)

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice
)
