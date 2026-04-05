"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach.
   And with Multimodal Input (e.g., images, top-down spatial map)."""
from typing import Any, Callable, List, ClassVar, Optional, Sequence, Tuple, Dict, Union
import os
import re
import json
from collections import defaultdict
from PIL import Image
from pydantic import Field, root_validator
from langchain.prompts import PromptTemplate
from langchain.prompts.base import check_valid_template
from langchain.schema.prompt import PromptValue
from langchain.schema.messages import HumanMessage, BaseMessage
from langchain.llms.openai import OpenAIChat, get_from_dict_or_env, completion_with_retry
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Generation, LLMResult
from langchain.schema.output import GenerationChunk
from langchain.utils import get_from_dict_or_env
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from .image_utils import encode_image



class MMStringPromptValue(PromptValue):
    """String prompt value."""

    text: str
    image_token: str = "<ImageHere>"
    images: Union[List[Image.Image], Image.Image, str, List[str], List[Tuple[Image.Image, str]]] = []
    class Config: arbitrary_types_allowed = True
    image_detail: str = "low"  # "low" or "high"

    def _get_content(self):
        content = []
        img_idx = 0
        txt_parts = re.split(r"(" + self.image_token + ")", self.text)
        for part in txt_parts:
            if part == "": continue
            if part == self.image_token and img_idx < len(self.images):
                if isinstance(self.images[img_idx], tuple) and len(self.images[img_idx]) == 2:
                    # the image with detail string is directly provided
                    cur_image, cur_detail = self.images[img_idx]
                else: cur_image, cur_detail = self.images[img_idx], self.image_detail
                content.append(encode_image(cur_image, detail=cur_detail))
                img_idx += 1
            else: # regular text part
                content.append({"type": "text", "text": part})
        assert img_idx == len(self.images), f"Text:\n{self.text}\n\n" + \
            f"Error: Number of image tokens ({img_idx}) in text " + \
            f"does not match number of images ({len(self.images)})."
        print(f"-- Finished formatting multimodal prompt with {img_idx} images --")
        return content

    def to_string(self) -> Union[str, List[Dict[str, str]]]:
        """Return prompt value as string."""
        return self._get_content()

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of Messages."""
        content = self._get_content()
        return [HumanMessage(content=content)]


class MultimodalPromptTemplate(PromptTemplate):
    """A prompt template that can handle multimodal inputs."""

    image_key: str = "images"
    image_token: ClassVar[str] = "<ImageHere>"
    image_detail: str = "low"  # "low" or "high"

    def format_prompt(self, **kwargs: Any) -> str:
        images = kwargs.pop(self.image_key, [])
        if not isinstance(images, list): images = [images]
        return MMStringPromptValue(
            text=self.format(**kwargs),
            images=images,
            image_token=self.image_token,
            image_detail=self.image_detail,
        )

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that template and input variables are consistent."""
        if values["validate_template"]:
            image_key = values.get("image_key", "images")
            if image_key in values["input_variables"]:
                valid_input_variables = set(values["input_variables"]) - {image_key}
                valid_input_variables = list(valid_input_variables)
            else: valid_input_variables = values["input_variables"]
            all_inputs = valid_input_variables + list(values["partial_variables"])
            check_valid_template(
                values["template"], values["template_format"], all_inputs
            )
        return values


class MultimodalOpenAI(OpenAIChat):
    """A multimodal version of OpenAI LLM that can handle multimodal prompts."""
    max_retries: int = 30
    """Maximum number of retries to make when generating."""

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        if self.streaming:
            generation: Optional[GenerationChunk] = None
            for chunk in self._stream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])

        messages, params = self._get_chat_params(prompts, stop)
        params = {**params, **kwargs}
        if "gpt-5" in self.model_name or "o1" in self.model_name or \
           "o3" in self.model_name or "o4" in self.model_name:
            stop = params.pop("stop", [])
        
        if "gpt-5.1" in self.model_name: 
            # turn off reasoning for gpt-5.1
            params["reasoning_effort"] = "medium"
            params["verbosity"] = "medium"

        if "qwen3-vl" in self.model_name:
            params["enable_thinking"] = True
            params["thinking_budget"] = 4096

        # add api key and base to params
        params.update({
            "api_key": self.openai_api_key,
            "api_base": self.openai_api_base,
        })
        
        full_response = completion_with_retry(
            self, messages=messages, run_manager=run_manager, **params
        )
        llm_output = {
            "token_usage": full_response["usage"],
            "model_name": self.model_name,
        }

        results = []
        # response_text = full_response["choices"][0]["message"]["content"]
        for choice in full_response["choices"]:
            response_text = choice["message"]["content"]
            if "gpt-5.1" in self.model_name: pass
            elif "gpt-5" in self.model_name or "o1" in self.model_name or \
            "o3" in self.model_name or "o4" in self.model_name:
                for sep in stop:
                    if sep in response_text:
                        response_text = response_text.split(sep)[0]
                        break
            results.append([Generation(text=response_text, 
                                      generation_info={"raw_response": choice["message"]["content"]})])
        
        return LLMResult(generations=results, llm_output=llm_output,)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        openai_api_base = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        openai_proxy = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        openai_organization = get_from_dict_or_env(
            values, "openai_organization", "OPENAI_ORGANIZATION", default=""
        )
        try:
            import openai

            openai.api_key = openai_api_key
            if openai_api_base:
                openai.api_base = openai_api_base
            if openai_organization:
                openai.organization = openai_organization
            if openai_proxy:
                openai.proxy = {"http": openai_proxy, "https": openai_proxy}  # type: ignore[assignment]  # noqa: E501
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        return values


class MLLMChain(LLMChain):
    """A multimodal LLM chain that can handle multimodal prompts."""
    return_final_only = False
    token_usage_save_path: Optional[str] = None
    cur_token_usage: Optional[Dict[str, Any]] = None

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[str, Dict[str, str]]:
        response = self.generate([inputs], run_manager=run_manager)
        
        if self.token_usage_save_path is not None:
            os.makedirs(os.path.dirname(self.token_usage_save_path), exist_ok=True)
            with open(self.token_usage_save_path, "a") as f: # save as jsonl
                f.write(json.dumps(response.llm_output, ensure_ascii=False) + "\n")
        # convert to standard dict for easier usage later
        self.cur_token_usage = json.loads(json.dumps(response.llm_output, ensure_ascii=False))
        
        outputs = self.create_outputs(response)
        if len(outputs) == 1: 
            return outputs[0]
        else: 
            output = defaultdict(list)
            for res in outputs:
                for k, v in res.items(): output[k].append(v)
            return output
