# sim11ah/constants.py

class FrameType:
    # Data
    DATA = "DATA"
    NULL_DATA = "NULL_DATA"

    # Control
    ACK = "ACK"
    RTS = "RTS"
    CTS = "CTS"
    CF_END = "CF_END"

    BLOCK_ACK = "BLOCK_ACK"
    BLOCK_ACK_REQ = "BLOCK_ACK_REQ"

    # Optional short aliases
    BA = BLOCK_ACK
    BAR = BLOCK_ACK_REQ

    PS_POLL = "PS_POLL"

    # Management
    BEACON = "BEACON"
    PROBE_REQ = "PROBE_REQ"
    PROBE_RESP = "PROBE_RESP"
    AUTH = "AUTH"
    ASSOC_REQ = "ASSOC_REQ"
    ASSOC_RESP = "ASSOC_RESP"
    DEAUTH = "DEAUTH"

    # 802.11ah / RAW related
    TIM = "TIM"
    DTIM = "DTIM"
    RAW_TRIGGER = "RAW_TRIGGER"
    NDP_ANNOUNCE = "NDP_ANNOUNCE"

    # TWT (Target Wake Time) negotiation, IEEE 802.11ah-2016 / 802.11-2020
    # Section 9.4.2.198-199 (TWT element carried in these mgmt frames)
    TWT_SETUP_REQ = "TWT_SETUP_REQ"
    TWT_SETUP_RESP = "TWT_SETUP_RESP"


class MacState:
    IDLE = "IDLE"
    BACKOFF = "BACKOFF"
    TX = "TX"
    RX = "RX"
    CCA = "CCA"
    DEFER = "DEFER"
    NAV = "NAV"
    WAIT_ACK = "WAIT_ACK"
    WAIT_CTS = "WAIT_CTS"
    RETRY = "RETRY"
    RAW_SLEEP = "RAW_SLEEP"
    RAW_WAIT = "RAW_WAIT"
    UNKNOWN = "UNKNOWN"