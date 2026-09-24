import argparse
import asyncio

import h5py
import numpy as np
import torch

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from nomad.tensor_codac import deserialize_tensor, serialize_tensor

from morph_pde.nomad_tool import MORPHTool


def load_pli_t0(input_h5: str) -> torch.Tensor:
    """Load the physical PLI t=0 field exactly as standalone inference does."""
    with h5py.File(input_h5, "r") as f:
        frame = f["t0_fields/av_density"][0, 0, :, :].astype(
            np.float32
        )

    frame = np.nan_to_num(
        frame,
        nan=0.0,
    )

    # Tool schema excludes batch:
    # (T,F,C,D,H,W)
    return torch.from_numpy(frame)[
        None,
        None,
        None,
        None,
        :,
        :,
    ]


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-h5",
        required=True,
    )
    parser.add_argument(
        "--model-dir",
        default="models/morph-s-pli",
        help=(
            "Directory containing model.pth, config.yaml, "
            "and normalization.npy"
        ),
    )
    parser.add_argument(
        "--mcp-url",
        default="http://localhost:8181/mcp",
    )
    parser.add_argument(
        "--tool-name",
        default="morph_s_pli",
        help="MCP tool name exposed by NOMAD.",
    )
    parser.add_argument(
        "--device",
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # -------------------------------------------------
    # 1. Real PLI input in physical units
    # -------------------------------------------------
    state = load_pli_t0(args.input_h5)

    print(
        "Physical input shape:",
        tuple(state.shape),
    )
    print(
        "Physical input range:",
        float(state.min()),
        float(state.max()),
    )

    # -------------------------------------------------
    # 2. Load config-driven MORPH-S + normalization
    # -------------------------------------------------
    direct_tool = MORPHTool.from_pretrained(
        args.model_dir,
        device=device,
    )

    if not hasattr(direct_tool.fm, "mean"):
        raise RuntimeError(
            "Normalization wrapper was not installed."
        )

    print(
        "Normalization mean:",
        direct_tool.fm.mean.detach().cpu().item(),
    )
    print(
        "Normalization variance:",
        direct_tool.fm.scale.detach().cpu().item(),
    )

    # -------------------------------------------------
    # 3. Manual reference
    # -------------------------------------------------
    x_batched = state.unsqueeze(0).to(device)

    mean = direct_tool.fm.mean
    variance = direct_tool.fm.scale
    bare_morph = direct_tool.fm.fm

    with torch.no_grad():
        x_norm = (
            x_batched - mean
        ) / variance

        _, _, prediction_norm = bare_morph(
            x_norm
        )

        reference_prediction = (
            prediction_norm * variance + mean
        )[0].detach().cpu()

    # -------------------------------------------------
    # 4. Wrapper inference
    # -------------------------------------------------
    with torch.no_grad():
        _, _, wrapped_prediction_batched = direct_tool.fm(
            x_batched
        )

    wrapped_prediction = (
        wrapped_prediction_batched[0]
        .detach()
        .cpu()
    )

    wrapped_diff = (
        reference_prediction - wrapped_prediction
    ).abs()

    print(
        "Reference vs wrapper max abs diff:",
        wrapped_diff.max().item(),
    )

    wrapper_match = torch.allclose(
        reference_prediction,
        wrapped_prediction,
        atol=1e-5,
        rtol=1e-5,
    )

    print(
        "Reference vs wrapper match:",
        wrapper_match,
    )

    if not wrapper_match:
        raise RuntimeError(
            "Normalization wrapper does not reproduce "
            "manual MORPH inference."
        )

    # -------------------------------------------------
    # 5. Same physical tensor through MCP
    # -------------------------------------------------
    encoded_state = serialize_tensor(
        state
    )

    async with streamable_http_client(
        args.mcp_url
    ) as streams:
        read_stream = streams[0]
        write_stream = streams[1]

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            await session.initialize()

            tools = await session.list_tools()
            available = [
                tool.name
                for tool in tools.tools
            ]

            print(
                "Available tools:",
                available,
            )

            if args.tool_name not in available:
                raise RuntimeError(
                    f"Expected MCP tool "
                    f"'{args.tool_name}', found {available}"
                )

            result = await session.call_tool(
                args.tool_name,
                {
                    "state": encoded_state,
                },
            )

            result_dict = result.model_dump(
                by_alias=True
            )

            is_error = result_dict.get(
                "isError",
                result_dict.get(
                    "is_error",
                    False,
                ),
            )

            if is_error:
                raise RuntimeError(
                    result_dict
                )

            structured = result_dict.get(
                "structuredContent",
                result_dict.get(
                    "structured_content"
                ),
            )

            mcp_prediction = deserialize_tensor(
                structured["prediction"]
            ).cpu()

    # -------------------------------------------------
    # 6. Compare manual reference and MCP
    # -------------------------------------------------
    print(
        "Reference output shape:",
        tuple(reference_prediction.shape),
    )
    print(
        "MCP output shape:",
        tuple(mcp_prediction.shape),
    )

    difference = (
        reference_prediction - mcp_prediction
    ).abs()

    print(
        "Reference vs MCP max abs diff:",
        difference.max().item(),
    )
    print(
        "Reference vs MCP mean abs diff:",
        difference.mean().item(),
    )

    mcp_match = torch.allclose(
        reference_prediction,
        mcp_prediction,
        atol=1e-5,
        rtol=1e-5,
    )

    print(
        "Reference vs MCP match:",
        mcp_match,
    )

    if not mcp_match:
        raise RuntimeError(
            "MCP/NOMAD prediction does not match "
            "direct PLI inference."
        )


if __name__ == "__main__":
    asyncio.run(main())
