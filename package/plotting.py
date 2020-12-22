# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import altair as alt
import altair_viewer


def candlestick(source, width=800, height=500, view=True, lines=None):
    """
    Function to generate a interactive candlestick chart for visualization of
    the time series of a financial source.

    Parameters:
    ===================
    :param source: pd.DataFrame: Time series DataFrame containing
                                 OLHCV values.
    :param width: int: The width of the chart.
    :param height: int: The height of the chart.
    :param view: bool: If True, it will return a URL to visualize the chart.
    :param lines: dict: Containing as keys the name of the columns and as
                        values the colors of the lines.

    Return:
    ===================
    The function returns a URL where the interactive chart will be displayed.
    """
    source.reset_index(inplace=True)

    # defining colors for the candlesticks
    open_close_color = alt.condition("datum.open <= datum.close",
                                     alt.value("#06982d"),
                                     alt.value("#ae1325"))

    # creating the base for the candlestick's chart
    base = alt.Chart(source).encode(
        alt.X('index:T',
              axis=alt.Axis(
                  format='%Y/%m/%d',
                  labelAngle=-90,
                  title='Dates')),
        color=open_close_color,
    )

    # creating a line for highest and lowest
    rule = base.mark_rule().encode(
        alt.Y(
            'low:Q',
            scale=alt.Scale(zero=False),
            axis=alt.Axis(title='Prices', orient='right'),
        ),
        alt.Y2('high:Q')
    )

    # creating the candlestick's bars
    bar = base.mark_bar().encode(
        alt.Y('open:Q'),
        alt.Y2('close:Q')
    )

    # joining OLHC together
    chart = rule + bar

    # drawing line
    # !!!!!!!!!! need to fix the problem with the colors
    if lines is not None:
        for k, v in lines.items():
            chart += base.mark_line(
                color=v,
                opacity=0.3
            ).encode(
                y=alt.Y(k)
            )

    # adding tooltips, properties and interaction
    chart = chart.encode(
        tooltip=[alt.Tooltip('index:T', title='Date'),
                 alt.Tooltip('open', title='Open'),
                 alt.Tooltip('low', title='Low'),
                 alt.Tooltip('high', title='High'),
                 alt.Tooltip('close', title='Close'),
                 alt.Tooltip('volume', title='Volume')]
    ).properties(
        width=width,
        height=height,
        title=f'Candlestick visualization'
    ).interactive()

    # creating x-axis selections
    # !!!!!!!!!! it is jumping bar - fix later
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['index'], empty='none')

    # drawing a vertical rule at the location of the selection
    v_rule = alt.Chart(source).mark_rule(color='gray').encode(
        x='index:T', ).transform_filter(nearest)

    # adding nearest selection on candlestick's chart
    chart = chart.add_selection(
        nearest
    )
    # ##########

    if view is True:
        # altair_viewer.show(chart)
        altair_viewer.display(chart + v_rule)
