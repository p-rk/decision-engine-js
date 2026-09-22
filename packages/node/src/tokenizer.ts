import type { Tokenizer } from "@decision-engine/core";
import {
  AutoTokenizer,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

export class HuggingFaceTokenizer implements Tokenizer {
  private constructor(private readonly tokenizer: PreTrainedTokenizer) {}

  static async load(tokenizerDirectory: string): Promise<HuggingFaceTokenizer> {
    const tokenizer = await AutoTokenizer.from_pretrained(tokenizerDirectory, {
      local_files_only: true,
    });
    return new HuggingFaceTokenizer(tokenizer);
  }

  get clsTokenId(): number {
    const tokenId = this.tokenizer.convert_tokens_to_ids("[CLS]");
    if (!Number.isInteger(tokenId) || tokenId < 0) {
      throw new Error("Tokenizer does not define a valid [CLS] token");
    }
    return tokenId;
  }

  get maskToken(): string {
    return this.tokenizer.mask_token;
  }

  get maskTokenId(): number {
    return this.tokenizer.mask_token_id;
  }

  get padTokenId(): number {
    return this.tokenizer.pad_token_id;
  }

  get sepTokenId(): number {
    return this.tokenizer.sep_token_id;
  }

  encode(text: string): number[] {
    return this.tokenizer.encode(text, { add_special_tokens: false });
  }
}
